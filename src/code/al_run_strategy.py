import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset
from dataloaders.dataset_fetalhead import BaseDataSets_HC18 as BaseDataSets
# from dataloaders.dataset_fetalhead import BaseDataSets_ESTT as BaseDataSets
from utils.utils import get_strategy
from unext.archs import UNext as Unet2D


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--patch_size', type=list,  default=[448, 448],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1234, help='random seed')
parser.add_argument('--exp', type=str,
                    default='ACDC/Semi_Mamba_UNet', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='mambaunet', help='model_name')
# AL strategies
parser.add_argument('--al_strategy', type=str, default=None,
                    help='Active learning strategy')
parser.add_argument('--al_iter', type=int, default=1,
                    help='Active learning loop')
parser.add_argument('--labeled_num', type=int, default=5,
                    help='labeled data')
args = parser.parse_args()

def runner(args):
    strategy_name = args.al_strategy
    al_iter = args.al_iter
    labeled_slice = args.labeled_num
    periter = 4000  # HC18, 10%
    # periter = 2000  # HC18, 5%    
    # periter = 2800  # ESTT, 10%
    # periter = 1200  # ESTT, 5%    

    assert al_iter>0, "The AL iteration must be greater than 0."

    tensor_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.patch_size[0], args.patch_size[1]), transforms.InterpolationMode.NEAREST),
    ])    

    selector_model = Unet2D(in_chns=3, out_channels=2, img_size=448, num_classes=2)
    model_path = f'{args.root_path}/al/model/{args.exp}_{labeled_slice}/Unext/model1_iter_{al_iter*periter}.pth'
    selector_model.load_state_dict(torch.load(model_path, map_location="cuda:0"))
    print(f'load a pre-trained model with file {model_path}')

    pre_iternum = al_iter - 1
    if strategy_name is not None:
        # get previous labeled data list
        with open(args.root_path + f"/ssl/train/labeled_data/{strategy_name}/data_al_{pre_iternum}_{labeled_slice}.list", "r") as f1:
            pre_sample_label_list = f1.readlines()

        with open(args.root_path + f"/ssl/train/unlabeled_data/{strategy_name}/data_al_{pre_iternum}_{labeled_slice}.list", "r") as f2:
            pre_sample_unlabel_list = f2.readlines()

        pre_sample_label_list = [item.replace("\n", "") for item in pre_sample_label_list]
        pre_sample_unlabel_list = [item.replace("\n", "") for item in pre_sample_unlabel_list]

        db_train_labled = BaseDataSets(base_dir=args.root_path, split="train", num=labeled_slice, islabeled=True, 
                                               isAL=True, alNum=pre_iternum, stgrategy=strategy_name,
                                               transform=None, default_transform=tensor_transforms)

        db_train_unlabled = BaseDataSets(base_dir=args.root_path, split="train", islabeled=False, 
                                              isAL=True, num=labeled_slice, 
                                              alNum=pre_iternum, stgrategy=strategy_name, 
                                              transform=None, default_transform=tensor_transforms)

        if strategy_name == "HybridSampling":
            strategy = get_strategy(strategy_name)(label_dataset=db_train_labled, 
                                                   unlabel_dataset=db_train_unlabled, 
                                                   net=selector_model) # load strategy
        else:
            strategy = get_strategy(strategy_name)(unlabel_dataset=db_train_unlabled, 
                                                   net=selector_model) # load strategy
        query_images = strategy.query(labeled_slice).tolist()
        print(f"selected {len(query_images)} samples from unlabeled data")

        # update the labeled and unlabeled data lists
        new_unlable_images = list(set(pre_sample_unlabel_list) - set(query_images))
        print(f"Now, there are {len(new_unlable_images)} unlabeled samples for training")
        new_query_images = pre_sample_label_list + query_images
        print(f"Now, there are {len(new_query_images)} labeled samples for training")

        with open(args.root_path + f"/ssl/train/labeled_data/{strategy_name}/data_al_{al_iter}_{labeled_slice}.list", "w") as f3:
            for line in new_query_images:
                f3.write(f"{line}\n")

        with open(args.root_path + f"/ssl/train/unlabeled_data/{strategy_name}/data_al_{al_iter}_{labeled_slice}.list", "w") as f4:
            for line in new_unlable_images:
                f4.write(f"{line}\n")

        print(f'{strategy_name} strategy selected {labeled_slice} samples for AL iteration {al_iter}')
        print(f'new labeled and unlabeled data locate at {args.root_path}/ssl/train/')


if __name__ == "__main__":

    random.seed(args.seed * args.al_iter)
    np.random.seed(args.seed * args.al_iter)
    torch.manual_seed(args.seed * args.al_iter)
    torch.cuda.manual_seed(args.seed * args.al_iter)

    snapshot_path = "{}/al/model/{}_{}/{}".format(
        args.root_path, args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    runner(args)
