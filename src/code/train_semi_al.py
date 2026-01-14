# autodl-tmp/Mamba-UNet/code/train_fully_supervised_2D_ViT.py
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
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import transforms
import albumentations as A
from tqdm import tqdm
from config import get_config
from dataloaders import utils
from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler)
from dataloaders.dataset_fetalhead import BaseDataSets_HC18 as BaseDataSets
# from dataloaders.dataset_fetalhead import BaseDataSets_ESTT as BaseDataSets
# from networks.unet import UNet as Unet2D
from networks.archs import UNext as Unet2D
from networks.vision_transformer import SwinUnet as ViT_seg
from utils import losses, ramps #, metrics
from utils.dice import DSC
from utils.utils import get_strategy
from utils.metrics import dice_score
from utils.losses import ConstraLoss, ConstraLoss_AvgProj, info_nce_loss, hd_loss, softmax_kl_loss, softmax_mse_loss


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Semi_Mamba_UNet', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='mambaunet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[448, 448],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1234, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument(
    # '--cfg', type=str, default="../code/configs/vmamba_tiny.yaml", help='path to config file', )
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

# AL strategies
parser.add_argument('--al_strategy', type=str, default=None,
                    help='Active learning strategy')
parser.add_argument('--al_iter', type=int, default=1,
                    help='Active learning loop')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=8,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=3,
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
# print(config)

device_name = "cuda:1"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

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
    strategy_name = args.al_strategy
    al_iter = args.al_iter
    # al_iter_num = al_iter * max_iterations
    val_size = 50
    val_steps = int((labeled_slice * (al_iter + 1)))

    con_rampup = args.consistency_rampup
    # iter_num//100
    # rampup_rate =  int((labeled_slice / args.labeled_bs) * 400 / con_rampup)
    # rampup_rate = 20 * (al_iter + 1)  # SSL: 100; AL: 20, 40, 60, 80, 100 # OLD (31/Oct, 2025)
    if al_iter == 0:
        rampup_rate = 100  # SSL only
    else:
        rampup_rate = 20 * al_iter  # SSL: 100; AL: 20, 40, 60, 80, 100
    # max_iterations = rampup_rate * con_rampup  # 20 * 200 = 4000
    # max_epoch = max_iterations / (labeled_slice / args.labeled_bs)  # 4000 / (10/1) = 400

    # model1 = Unet2D(in_chns=3, class_num=args.num_classes).cuda()
    # model2 = Unet2D(in_chns=3, class_num=args.num_classes).cuda()

    model1 = Unet2D(in_chns=3,
                   out_channels=args.num_classes,
                   img_size=args.patch_size[0],
                   num_classes=args.num_classes).to(device)

    model2 = ViT_seg(config, 
                     img_size=args.patch_size[0],
                     num_classes=args.num_classes).to(device)
    model2.load_from(config, device_name)    
    
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
        transforms.Resize((args.patch_size[0], args.patch_size[1]), transforms.InterpolationMode.NEAREST),
    ])

    # 1st round AL iteration = 0
    # also, SSL training
    if strategy_name is not None and al_iter is not None:
        db_train_al_labled = BaseDataSets(base_dir=args.root_path, split="train", num=labeled_slice, islabeled=True, 
                                               isAL=True, alNum=al_iter, stgrategy=strategy_name,
                                               transform=tr_transforms, default_transform=tensor_transforms)
        
        # db_train = ConcatDataset([db_train, db_train_al_labled, db_train_al_labled_da])
        # db_train = ConcatDataset([db_train_labled, db_train_al_labled])
        # logging.info(f'lenght of AL labled train data: {len(db_train_al_labled)}, lenght of AL labeled train data with DA: {len(db_train_al_labled_da)},')
        logging.info(f'lenght of AL labled train data: {len(db_train_al_labled)}')

        db_train_al_unlabled = BaseDataSets(base_dir=args.root_path, split="train", num=labeled_slice, islabeled=False, 
                                              isAL=True, alNum=al_iter, stgrategy=strategy_name,
                                              transform=tr_transforms, default_transform=tensor_transforms)
        logging.info(f'lenght of AL unlabled train data: {len(db_train_al_unlabled)}')
        db_train = ConcatDataset([db_train_al_labled, db_train_al_unlabled])
    else:
        db_train_labled = BaseDataSets(base_dir=args.root_path, split="train", num=labeled_slice, islabeled=True, 
                                            isAL=True, stgrategy=strategy_name, 
                                            transform=tr_transforms, default_transform=tensor_transforms)
        db_train = db_train_labled
        logging.info(f'total lenght of labled train data: {len(db_train_labled)}')
        
    logging.info(f'lenght of labled train data: {len(db_train)}')
    print(f'total lenght of train data in total: {len(db_train)}')

    db_val = BaseDataSets(base_dir=args.root_path, split="val", transform=None, default_transform=tensor_transforms)
    # print(f'total lenght of val data: {len(db_val)}')
    logging.info(f'total lenght of val data: {len(db_val)}')
    
    total_slices = len(db_train)
    print("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice * (al_iter + 1)))
    labeled_idxs = list(range(0, labeled_slice * (al_iter + 1)))
    unlabeled_idxs = list(range(labeled_slice * (al_iter + 1), total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)
    
    # print("labeled_idxs: {}, unlabeled_idxs: {}".format(labeled_idxs, unlabeled_idxs))

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=1, pin_memory=False, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    
    model1.train()
    model2.train()

    if strategy_name is not None and al_iter > 0:
        # pre_iter_num = al_iter * 2000  # HC18, 5%
        pre_iter_num = al_iter * 4000  # HC18, 10%
        # pre_iter_num = al_iter * 2800  # ESTT, 10%
        # pre_iter_num = al_iter * 1200  # ESTT, 5%
        model1.load_state_dict(torch.load(os.path.join(snapshot_path, f'model1_iter_{pre_iter_num}.pth')))
        model2.load_state_dict(torch.load(os.path.join(snapshot_path, f'model2_iter_{pre_iter_num}.pth')))
        print(f"load model1 from {os.path.join(snapshot_path, f'model1_iter_{pre_iter_num}.pth')}")
        logging.info(f"load model1 from {os.path.join(snapshot_path, f'model1_iter_{pre_iter_num}.pth')}")
    
    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    time1 = int(time.time())
    writer = SummaryWriter(snapshot_path + f'/log_{time1}')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    evalue = DSC()  # metric to find best model
    iterator = tqdm(range(max_epoch), ncols=70)
    
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)

            unlabeled_volume_batch = volume_batch[args.labeled_bs:]
            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise
            
            outputs1, outputs1_emb = model1(volume_batch)
            outputs_soft1 = torch.softmax(outputs1, dim=1)

            outputs2, outputs2_emb = model2(volume_batch)  # B, 14*14, 768
            outputs_soft2 = torch.softmax(outputs2, dim=1)

            ema_outputs2, _ = model2(ema_inputs)

            consistency_loss = ConstraLoss_AvgProj(outputs1[args.labeled_bs:], ema_outputs2.detach(), ndf=128)
            
            consistency_weight = get_current_consistency_weight(iter_num // rampup_rate)
            
            loss1 = 0.5 * (ce_loss(outputs1[:args.labeled_bs], torch.squeeze(label_batch[:args.labeled_bs],1).long()) + 
                           dice_loss(outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs]))
            
            loss2 = 0.5 * (ce_loss(outputs2[:args.labeled_bs], torch.squeeze(label_batch[:args.labeled_bs],1).long()) + 
                           dice_loss(outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs]))
            
            pseudo_outputs1 = torch.argmax(outputs_soft1[args.labeled_bs:].detach(), dim=1, keepdim=False)
            pseudo_outputs2 = torch.argmax(outputs_soft2[args.labeled_bs:].detach(), dim=1, keepdim=False)

            pseudo_supervision1 = dice_loss(outputs_soft1[args.labeled_bs:], pseudo_outputs2.unsqueeze(1))
            pseudo_supervision2 = dice_loss(outputs_soft2[args.labeled_bs:], pseudo_outputs1.unsqueeze(1))
            
            model1_loss = loss1 + consistency_weight * pseudo_supervision1 + 0.5*consistency_loss
            model2_loss = loss2 + consistency_weight * pseudo_supervision2 + 0.5*consistency_loss
            
            # model1_loss = loss1 + consistency_weight * pseudo_supervision1 + 0.5*con1
            # model2_loss = loss2 + consistency_weight * pseudo_supervision2 + 0.5*con1 
            
            loss = model1_loss + model2_loss 

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()

            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/model1_loss',
                              model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss',
                              model2_loss, iter_num)
            logging.info('iteration %d : model1 loss : %f model2 loss : %f' % (
                iter_num, model1_loss.item(), model2_loss.item()))
            
            if iter_num > 0 and iter_num % val_steps == 0:  # do evaluation every epochs
                model1.eval()
                model2.eval()
                
                metric_list1 = 0.0
                metric_list2 = 0.0
                
                for i_batch, sampled_batch in enumerate(valloader):
                    volume_valbatch, label_valbatch = sampled_batch['image'], sampled_batch['label']
                    volume_valbatch, label_valbatch = volume_valbatch.to(device), label_valbatch.to(device)
                    label_valbatch = torch.squeeze(label_valbatch,1)
                    
                    # print(f'label_valbatch shape: {label_valbatch.shape}')
                    
                    preds_1, _ = model1(volume_valbatch)
                    preds_2, _ = model2(volume_valbatch)
                    
                    preds_1_soft = torch.softmax(preds_1, dim=1)
                    preds_2_soft = torch.softmax(preds_2, dim=1)
                    
                    
                    dsc1_i = dice_score(preds_1_soft, label_valbatch)
                    dsc2_i = dice_score(preds_2_soft, label_valbatch)
                    
                    metric_list1 += dsc1_i
                    metric_list2 += dsc2_i

                performance1 = metric_list1 / len(db_val)
                performance2 = metric_list2 / len(db_val)
                
                writer.add_scalar('info/model1_val_mean_dice', performance1, iter_num)
                writer.add_scalar('info/model2_val_mean_dice', performance2, iter_num)
            
                performance1 = performance1.item()
                performance2 = performance2.item()

                if performance1 > best_performance1 and iter_num > (2 * max_iterations / 3):
                    best_performance1 = performance1
                    if al_iter == 4:
                        save_mode_path = os.path.join(snapshot_path,'model1_iter_{}_dice_{}.pth'.format(iter_num, round(best_performance1, 4)))
                        save_best = os.path.join(snapshot_path,'{}_best_model1.pth'.format(args.model))
                        torch.save(model1.state_dict(), save_mode_path)
                        torch.save(model1.state_dict(), save_best)
                    
                if performance2 > best_performance2 and iter_num > (2 * max_iterations / 3):
                    best_performance2 = performance2
                    # if al_iter == 4:
                    #     save_mode_path = os.path.join(snapshot_path,'model2_iter_{}_dice_{}.pth'.format(iter_num, round(best_performance2, 4)))
                    #     save_best = os.path.join(snapshot_path,'{}_best_model2.pth'.format(args.model))
                    #     torch.save(model2.state_dict(), save_mode_path)
                    #     torch.save(model2.state_dict(), save_best)

                logging.info('iteration %d : model1_mean_dice : %f ' % (iter_num, performance1))
                logging.info('iteration %d : model2_mean_dice : %f ' % (iter_num, performance2))
                logging.info('best performances so far : model_1 - %f, model_2 - %f ' % (best_performance1, best_performance2))
                
                model1.train()
                model2.train()
            
            # if iter_num % 250 == 0:
                # image = volume_batch[0, ...] * 255
                # image_save = image.type(torch.int64)
                # print('-'*20)
                # print(f'image shape: {image.shape}')
                # print(f'max {image.max()}')
                # print(f'min {image.min()}')
                # cv2.imwrite(f"{args.root_path}/ssl/assets_unet_al_{labeled_slice}/image_{iter_num}.png",np.transpose(image_save.data.cpu().numpy(),(1,2,0))) 
                # writer.add_image('train/Image', image_save, iter_num)
                
                # outputs = torch.softmax(outputs1, dim=1)
                # outputs = torch.argmax(torch.softmax(outputs1, dim=1), dim=1, keepdim=True)
                # img1_outputs = np.squeeze(outputs[0]) * 100
                # test_output = torch.softmax(outputs1, dim=1)
                # img_tensor = test_output[0, ...] * 255
                # print('-'*20)
                # print(f'outputs1 shape: {outputs1.shape}')
                # print(f'max {outputs1.max()}')
                # print(f'min {outputs1.min()}')
                # print(f'outputs shape: {outputs[0].shape}')
                # print(f'max {outputs[0].max()}')
                # print(f'min {outputs[0].min()}')
                # print(f'test_output shape: {test_output.shape}')
                # print(f'max {test_output.max()}')
                # print(f'min {test_output.min()}')
                # print(f'img_tensor shape: {img_tensor.shape}')
                # print(f'max r: {img_tensor[0].max()}, g: {img_tensor[0].max()}, b:{img_tensor[2].max()} ')
                # print(f'min r: {img_tensor[0].min()}, g: {img_tensor[0].min()}, b:{img_tensor[2].min()} ')
                # cv2.imwrite(f"{args.root_path}/ssl/assets_unet_al_{labeled_slice}/outputs1_{iter_num}.png",img1_outputs.data.cpu().numpy())
                # writer.add_image('train/model1_Prediction', outputs[0, ...] * 100, iter_num, dataformats='CHW')
                
                # outputs = torch.softmax(outputs2, dim=1)
                # outputs = torch.argmax(torch.softmax(outputs2, dim=1), dim=1, keepdim=True) 
                # img2_outputs = np.squeeze(outputs[0]) * 100
                # test_output = torch.softmax(outputs2, dim=1)
                # img_tensor = test_output[0, ...] * 255
                # print('-'*20)
                # print(f'outputs2 shape: {outputs2.shape}')
                # print(f'max {outputs2.max()}')
                # print(f'min {outputs2.min()}')
                # print(f'outputs shape: {outputs[0].shape}')
                # print(f'max {outputs[0].max()}')
                # print(f'min {outputs[0].min()}')
                # print(f'test_output shape: {test_output.shape}')
                # print(f'max {test_output.max()}')
                # print(f'min {test_output.min()}')
                # print(f'img_tensor shape: {img_tensor.shape}')
                # print(f'max r: {img_tensor[0].max()}, g: {img_tensor[0].max()}, b:{img_tensor[2].max()} ')
                # print(f'min r: {img_tensor[0].min()}, g: {img_tensor[0].min()}, b:{img_tensor[2].min()} ')
                # cv2.imwrite(f"{args.root_path}/ssl/assets_unet_al_{labeled_slice}/outputs2_{iter_num}.png",img2_outputs.data.cpu().numpy())
                # writer.add_image('train/model2_Prediction', outputs[0, ...] * 100, iter_num, dataformats='CHW')
                
                # labs = label_batch[0, ...].unsqueeze(0) * 100
                # labs = label_batch[0, ...]
                # labs_tensor = labs * 100
                # print('-'*20)
                # print(f'label_batch shape: {label_batch.shape}')
                # print(f'labs shape: {labs.shape}')
                # print(f'max {labs.max()}')
                # print(f'min {labs.min()}')
                # print(f'max r: {labs[0].max()}, g: {labs[0].max()}, b:{labs[2].max()} ')
                # print(f'min r: {labs[0].min()}, g: {labs[0].min()}, b:{labs[2].min()} ')
                # cv2.imwrite(f"{args.root_path}/ssl/assets_unet_al_{labeled_slice}/labs_{iter_num}.png",np.transpose(labs_tensor.data.cpu().numpy(),(1,2,0)))
                # writer.add_image('train/GroundTruth', labs_tensor, iter_num, dataformats='CHW')
                
            # if iter_num % 5000 == 0:
            current_max_iternum = (al_iter + 1) * 4000  # HC18, 10%
            # current_max_iternum = (al_iter + 1) * 2000  # HC18, 5%
            # current_max_iternum = (al_iter + 1) * 2800  # ESTT, 10%
            # current_max_iternum = (al_iter + 1) * 1200  # ESTT, 5%
            if iter_num == max_iterations:
                save_mode_path = os.path.join(
                    snapshot_path, 'model1_iter_' + str(current_max_iternum) + '.pth')
                torch.save(model1.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))

                save_mode_path = os.path.join(
                    snapshot_path, 'model2_iter_' + str(current_max_iternum) + '.pth')
                torch.save(model2.state_dict(), save_mode_path)
                logging.info("save model2 to {}".format(save_mode_path))
                
            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    

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

    snapshot_path = "{}/al/model/{}_{}/{}".format(
        args.root_path, args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)    