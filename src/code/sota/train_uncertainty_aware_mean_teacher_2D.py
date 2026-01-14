import argparse
import logging
import os
import random
import shutil
import sys
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from utils.metrics import dice_score
from config import get_config
from networks.archs import UNext as Unet2D
from dataloaders import utils
import albumentations as A
# from dataloaders.dataset_fetalhead import BaseDataSets_HC18 as USDataSets
from dataloaders.dataset_fetalhead import BaseDataSets_ESTT as USDataSets
from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Uncertainty_Aware_Mean_Teacher_ViT', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[448, 448],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument(
    '--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()
config = get_config(args)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    labeled_slice = args.labeled_num
    val_size = 50

    con_rampup = args.consistency_rampup
    # iter_num//100
    rampup_rate =  int((labeled_slice / args.labeled_bs) * 400 / con_rampup)

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes).to(device)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model
    
    model = Unet2D(in_chns=3,
                   out_channels=args.num_classes,
                   img_size=args.patch_size[0],
                   num_classes=args.num_classes).to(device)

    ema_model = Unet2D(in_chns=3,
                   out_channels=args.num_classes,
                   img_size=args.patch_size[0],
                   num_classes=args.num_classes).to(device)
    for param in ema_model.parameters():
        param.detach_()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    tr_transforms = A.Compose(
        [
            A.Rotate(limit=20, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2),contrast_limit=(-0.5, 0.5)),
            A.Blur(blur_limit=(3, 3), p=0.3),
            A.GaussNoise(std_range=(0.05, 0.1), p=0.3),
         ],
    )
    tensor_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.patch_size[0], args.patch_size[1])),
    ])  
    db_train_labled = USDataSets(base_dir=args.root_path, split="train", num=labeled_slice, islabeled=True, 
                                        transform=tr_transforms, default_transform=tensor_transforms) 
    db_train_unlabled = USDataSets(base_dir=args.root_path, split="train", num=labeled_slice, islabeled=False, 
                                           transform=None, default_transform=tensor_transforms)
    db_train = ConcatDataset([db_train_labled, db_train_unlabled])
    db_val = USDataSets(base_dir=args.root_path, split="val", num=val_size, 
                        transform=None, default_transform=tensor_transforms)
    total_slices = len(db_train)
    logging.info("Total silices is: {}, labeled slices is: {}".format(total_slices, len(db_train_labled)))
    logging.info("Val silices is: {}".format(len(db_val)))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    time1 = int(time.time())
    writer = SummaryWriter(snapshot_path + f'/log_{time1}')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]

            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise

            outputs, _ = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            with torch.no_grad():
                ema_output, _ = ema_model(ema_inputs)
            T = 8
            _, _, w, h = unlabeled_volume_batch.shape
            volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            preds = torch.zeros([stride * T, num_classes, w, h]).to(device)
            for i in range(T//2):
                ema_inputs = volume_batch_r + \
                    torch.clamp(torch.randn_like(
                        volume_batch_r) * 0.1, -0.2, 0.2)
                with torch.no_grad():
                    preds[2 * stride * i:2 * stride * (i + 1)], _ = ema_model(ema_inputs)
            preds = F.softmax(preds, dim=1)
            preds = preds.reshape(T, stride, num_classes, w, h)
            preds = torch.mean(preds, dim=0)
            uncertainty = -1.0 * torch.sum(preds*torch.log(preds + 1e-6), dim=1, keepdim=True)

            loss_ce = ce_loss(outputs[:args.labeled_bs], torch.squeeze(label_batch[:args.labeled_bs],1).long())
            loss_dice = dice_loss(outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs])

            supervised_loss = 0.5 * (loss_dice + loss_ce)
            consistency_weight = get_current_consistency_weight(iter_num//rampup_rate)
            consistency_dist = losses.softmax_mse_loss(outputs[args.labeled_bs:], ema_output)
            threshold = (0.75+0.25*ramps.sigmoid_rampup(iter_num, max_iterations))*np.log(2)
            mask = (uncertainty < threshold).float()
            consistency_loss = torch.sum(mask*consistency_dist)/(2*torch.sum(mask)+1e-16)

            loss = supervised_loss + consistency_weight * consistency_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)
            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            '''
            if iter_num % 250 == 0:
                image = volume_batch[0, ...] * 255
                image_save = image.type(torch.int64)
                cv2.imwrite(f"{args.root_path}/ssl/assets_MT_unext_{labeled_slice}/image_{iter_num}.png",np.transpose(image_save.data.cpu().numpy(),(1,2,0))) 
                writer.add_image('train/Image', image_save, iter_num)

                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                img1_outputs = np.squeeze(outputs[0]) * 100
                cv2.imwrite(f"{args.root_path}/ssl/assets_MT_unext_{labeled_slice}/outputs1_{iter_num}.png",img1_outputs.data.cpu().numpy())
                writer.add_image('train/Prediction', outputs[0, ...] * 100, iter_num, dataformats='CHW')

                labs_tensor = label_batch[0, ...] * 100
                labs = label_batch[1, ...].unsqueeze(0) * 50
                cv2.imwrite(f"{args.root_path}/ssl/assets_MT_unet_{labeled_slice}/labs_{iter_num}.png",np.transpose(labs_tensor.data.cpu().numpy(),(1,2,0)))
                writer.add_image('train/GroundTruth', labs_tensor, iter_num, dataformats='CHW')
            '''
            
            if iter_num > 0 and iter_num % (labeled_slice / args.labeled_bs) == 0:
                model.eval()
                metric_list = 0.0

                for i_batch, sampled_batch in enumerate(valloader):
                    volume_valbatch, label_valbatch = sampled_batch['image'], sampled_batch['label']
                    volume_valbatch, label_valbatch = volume_valbatch.to(device), label_valbatch.to(device)
                    label_valbatch = torch.squeeze(label_valbatch,1)

                    preds, _ = model(volume_valbatch)
                    preds_soft = torch.softmax(preds, dim=1)
                    # calculate all classes
                    dsc_i = dice_score(preds_soft, label_valbatch)
                    metric_list += dsc_i

                performance = metric_list / len(db_val)
                writer.add_scalar('info/model_val_mean_dice', performance, iter_num)
                performance = performance.item()

                if performance > best_performance and iter_num > (max_iterations / 2):
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model1_iter_{}_dice_{}.pth'.format(
                                                      iter_num, 
                                                      round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,'{}_best_model1.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info('iteration %d : model1_mean_dice : %f ' % (iter_num, performance))
                model.train()

            if iter_num == max_iterations:
                save_mode_path = os.path.join(snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "{}/ssl/model/{}_{}/{}".format(
        args.root_path, args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
