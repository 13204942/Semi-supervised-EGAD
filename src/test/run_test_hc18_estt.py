#!/usr/bin/env python
# coding: utf-8


import numpy as np
from os.path import isfile
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import cv2
import datetime
import logging
import sys
import argparse

from torchvision import transforms
# from unet.model import model
from unext.model import model
from utils.metrics import DSC, HD, calculate_metric_percase
# from resnetunet.model import model
# from effunet.model import model
# from attunet.model import model
# from mitunet.model import model
from swinunet.model import model as vit_seg
from medpy.metric.binary import hd95, asd

parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int,  default=2, help='output channel of network')
parser.add_argument('--model_name', type=str,  default="unext", help='test model name: swinunet or unext')


def dsc_fh(y_pred, y_truth, num_classes=2):
    """
    :param y_pred: (BS,3,336,544)
    :param y_truth: (BS,336,544)
    :return:
    """
    smooth = 1e-6
    y_pred_f = F.one_hot(y_pred.long(), num_classes)  # (BS,336,544,3)   
    y_pred_f = torch.flatten(y_pred_f, start_dim=0, end_dim=1)   # (-1,3)

    y_truth_f = F.one_hot(y_truth.long(), num_classes)  # (BS,336,544,3)
    y_truth_f = torch.flatten(y_truth_f, start_dim=0, end_dim=1)  # (-1,3)

    dice1 = (2. * ((y_pred_f[:, 1:] * y_truth_f[:, 1:]).sum()) + smooth) / (
            y_pred_f[:, 1:].sum() + y_truth_f[:, 1:].sum() + smooth)

    return dice1


if __name__ == "__main__":
    args = parser.parse_args()

    # UNeXt
    # test_model = model()
    # Swin-Unet
    # test_model = vit_seg()

    # UNeXt
    # test_model = model(num_classes=3)
    # Swin-Unet
    # test_model = vit_seg(num_classes=3)

    if args.model_name == 'unext':
        test_model = model(num_classes=args.num_classes)
    else:
        test_model = vit_seg(num_classes=args.num_classes)
    ################################################################################################
    # ### ES-TT

    proj_name = 'ESTT'
    root_dir = '/mnt/storage/fangyijie/ESTT'
    root_data = '/mnt/storage/fangyijie/ESTT/'

    # SOTA
    # MT 17, 34
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/MT_Unext_17/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/MT_Unext_34/Unext', iternum=None)

    # CPS 17, 34
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/CPS_Unext_17/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/CPS_Unext_34/Unext', iternum=None)

    # ICT 17, 34
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/ICT_Unext_17/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/ICT_Unext_34/Unext', iternum=13600)

    # DAN 17, 34
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/DAN_Unext_17/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/DAN_Unext_34/Unext', iternum=None)

    # UAMT 17, 34
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/UAMT_Unext_17/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/UAMT_Unext_34/Unext', iternum=None)

    # DCT 17, 34
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/DCT_Unext_17/Unext', iternum=6800)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/DCT_Unext_34/Unext', iternum=None)

    # CT-CT 17, 34
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/CTCT_Unext_17/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/CTCT_Unext_34/Unext', iternum=None)

    # PCPCS 17, 34
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/PCPCS_Unext_17/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/PCPCS_Unext_34/Unext', iternum=None)

    # full supervision models
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/Fullsup_Unet_500/unet', iternum=20000, ssl=False)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/Fullsup_Unext_342/unext', iternum=None, ssl=False)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/Fullsup_Unext_17/unext', iternum=None, ssl=False)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/Fullsup_Unext_34/unext', iternum=None, ssl=False)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/Fullsup_SwinUnet_342/swinunet', iternum=None, ssl=False)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/Fullsup_SwinUnet_17/swinunet', iternum=None, ssl=False)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/Fullsup_SwinUnet_34/swinunet', iternum=None, ssl=False)

    # Ours (UNeXt x 2)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/Semi_Unext_Unext_Consist_proj_AblationStudy_17/Unext', iternum=None)
    # Ours (UNeXt x 2) + L_semi
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/Semi_Unext_Unext_Consist_proj_AS_NoCon_17/Unext', iternum=None)

    # CPS (UNeXt x 2) + ISBI infonce + f2_KL_labeled (no background)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/Semi_Unext_Unext_17/Unext', iternum=10000)

    # CPS (UNeXt + Swin-unet) + contrastive loss + ema teacher
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/Semi_Swin_Unext_ema_17/Unext', iternum=10000)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/Semi_Swin_Unext_ema_34/Unext', iternum=20000)


    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/Semi_Swin_Unext_Consist_proj_17/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/Semi_Swin_Unext_Consist_proj_34/Unext', iternum=None)

    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/Semi_Swin_Unext_Consist_proj_v2_17/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/Semi_Swin_Unext_Consist_proj_v2_34/Unext', iternum=None)

    # Ours Ablation Study in journal
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/Semi_Swin_Unext_Consist_proj_v2_Ablation_param_34/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/Semi_Swin_Unext_Consist_proj_v2_Ablation_param_34/Unext', iternum=None)

    # active learning
    loaded_model = test_model.load(f'{root_dir}/al/model/ESTT/Semi_Unext_Swin_AL_3/Unext/RandomSampler', iternum=None)    

    ################################################################################################
    # ### HC18

    # proj_name = 'HC18'
    # root_dir = '/mnt/storage/fangyijie/HC18'
    # root_data = '/mnt/storage/fangyijie/HC18/'

    # SOTA
    # MT 25, 50
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/MT_Unext_25/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/MT_Unext_50/Unext', iternum=None)

    # CPS 25, 50
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/CPS_Unext_25/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/CPS_Unext_50/Unext', iternum=None)

    # ICT 25, 50
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/ICT_Unext_25/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/ICT_Unext_50/Unext', iternum=None)

    # DAN 25, 50
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/DAN_Unext_25/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/DAN_Unext_50/Unext', iternum=None)

    # UAMT 25, 50
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/UAMT_Unext_25/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/UAMT_Unext_50/Unext', iternum=None)

    # DCT 25, 50
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/DCT_Unext_25/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/DCT_Unext_50/Unext', iternum=20000)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/DCT_Unext_50/Unext', model_name='model1_iter_14900_dice_0.9684.pth')

    # CT-CT 25, 50
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/CTCT_Unext_25/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/CTCT_Unext_25/Unext', model_name='model1_iter_9100_dice_0.9645.pth')
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/CTCT_Unext_50/Unext', iternum=17400)

    # PCPCS 25, 50
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/PCPCS_Unext_25/Unext', iternum=10000)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/PCPCS_Unext_25/Unext', model_name='model1_iter_9800_dice_0.9617.pth')
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/PCPCS_Unext_50/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/PCPCS_Unext_50/Unext', iternum=20000)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/PCPCS_Unext_50/Unext', model_name='model1_iter_19800_dice_0.9747.pth')
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/PCPCS_Unext_25/Unext', subfolder='/old', model_name='model1_iter_9800_dice_0.9617.pth')

    # DSTCT 25, 50
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/DSTCT_Unext_25/Unext', iternum=10000)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/DSTCT_Unext_50/Unext', iternum=None)

    # ssl models 50
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Unet_Unet_50/unet')

    # ssl models 5, iternum=2000
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Unet_Unet_5/unet', iternum=2000)

    # ssl models 10, iternum=4000
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Unet_Unet_10/unet', iternum=4000)

    # ssl models 15, iternum=6000
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Unet_Unet_15/unet', iternum=6000)

    # ssl models 20, iternum=8000
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Unet_Unet_20/unet', iternum=8000)

    # ssl models 25, iternum=10000
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Unet_Unet_25/unet', iternum=10000)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_25/Unext_nosam', iternum=10000)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_25/Unext_nosam_noctraloss', iternum=10000)

    # CPS + original infonce
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_25/Unext_nosam_infonce_orig', iternum=10000)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_25/Unext_nosam_infonce_orig_argmax', iternum=10000)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_25/Unext_nosam_infonce_256', iternum=10000)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_25/Unext', iternum=10000)

    # CPS + ISBI infonce
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_25/Unext_nosam_infonce_isbi', iternum=10000)
    # CPS + ISBI infonce + f2_KL_labeled
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_25/Unext_nosam_infonce_isbi_f2kl_labeled', iternum=10000)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_NCE_KL_25/Unext', iternum=10000)
    # CPS + ISBI infonce + f2_KL_labeled
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_25/Unext_nosam_infonce_isbi_f2kl_labeled', iternum=10000)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_KL_25/Unext', iternum=10000)
    # CPS + FocalLoss
    # /home/fangyijie/ssl_us/SSL_Cervical_Segmentation/src/code/train_semi_swin_unext_KL.py
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_FocalLoss_25/Unext', iternum=10000)
    # CPS+ ConsistencyLoss (KL)
    # /home/fangyijie/ssl_us/SSL_Cervical_Segmentation/src/code/train_semi_swin_unext_KL.py
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_ConsisLoss_25/Unext', iternum=None)
    # CPS+ ConsistencyLoss (MSE)
    # /home/fangyijie/ssl_us/SSL_Cervical_Segmentation/src/code/train_semi_swin_unext_consist.py
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_Consist_25/Unext', iternum=None)
    # CPS+ ConstraLoss(outputs1[args.labeled_bs:], ema_outputs2)
    # /home/fangyijie/ssl_us/SSL_Cervical_Segmentation/src/code/train_semi_swin_unext_consist.py
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_Consist_v2_25/Unext', iternum=None)
    # CPS+ ConsistencyLoss (MSE)
    # /home/fangyijie/ssl_us/SSL_Cervical_Segmentation/src/code/train_semi_swin_unext_consist.py
    # torch.mean(softmax_mse_loss(outputs1[args.labeled_bs:], ema_outputs2.detach()))
    # torch.mean(softmax_mse_loss(outputs1[args.labeled_bs:], ema_outputs2))
    # CPS+ ConsistencyLoss (projector)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_Consist_MSE_25/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_Consist_MSE_50/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_Consist_proj_25/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_Consist_proj_50/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_Consist_proj_50/Unext', iternum=22500)

    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_Consist_proj_v2_25/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_Consist_proj_v2_test_25/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_Consist_proj_v2_test_2_25/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_Consist_proj_v3_25/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_Consist_proj_v2_nolcon_25/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_Consist_proj_v2_nolccd_25/Unext', iternum=None)

    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_Consist_proj_v2_50/Unext', iternum=None)


    # Ours Ablation Study (SwinUnet, UNeXt) + L_semi
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_Consist_proj_AS_NoCon_25/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_Consist_proj_v2_Ablation_50/Unext', iternum=None)
    # Ours Ablation Study (UNeXt x 2)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Unext_Unext_Consist_proj_AS_25/Unext', iternum=10000)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_Consist_proj_v2_Ablation_param_50/Unext', iternum=None)

    # #labeled = 50
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_50/Unext_ours', iternum=20000)


    # CPS + ISBI infonce + f2_KL_UL_L
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_25/Unext_nosam_infonce_isbi_f2kl_UL', iternum=10000)

    # CPS + SAM (KD SAM -> f_1, SAM -> f_2)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_SAM_25/Unext_freezeSAM_KD_SAMf1_SAMf2', iternum=10000)

    # CPS + SAM (KD SAM -> f_2, f_2 -> f_1)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_SAM_25/Unext_freezeSAM_KD_SAM_f2_f1', iternum=10000)

    # CPS + SAM (KD SAM -> f_1)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_SAM_KL_25/Unext', iternum=None)

    # CPS
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_25/Unext_CPS', iternum=10000)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_25/Unext_nosam_noctraloss', iternum=10000)

    # CPS + SAM (KD SAM -> f_1)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_SAM_25/Unext_freezeSAM_KD_SAM_f1', iternum=10000)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_SAM_25/Unext_freezeSAM_KD_SAM_f1_256', iternum=10000)

    # CPS + SAM (KD SAM -> f_1) + infonce
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_SAM_NCE_25/Unext_NCE_KD_SAM_f1', iternum=10000)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_SAM_NCE_25/Unext_NCE_KD_SAM_f1_256', iternum=10000)

    # CPS 256x256
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_25/Unext_CPS_256', iternum=10000)

    # ssl models 50, iternum=20000
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Unet_Unet_50/unet', iternum=20000)

    # ssl models 25
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Unet_Unet_25/unet')

    # Active Learning
    # ssl models 5
    # iternum = 2000, 4000, 6000, 8000, 10000
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Unet_Unet_AL_5/unet', iternum=4000)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Unet_Unet_AL_5/unet', iternum=6000)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Unet_Unet_AL_5/unet', iternum=8000)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Unet_Unet_AL_5/unet', iternum=10000)

    # full supervision models
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Fullsup_Unet_500/unet', iternum=20000, ssl=False)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Fullsup_Unext_500/unext', iternum=50000, ssl=False)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Fullsup_Unext_25/unext', iternum=10000, ssl=False)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Fullsup_Unext_50/unext', iternum=20000)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Fullsup_SwinUnet_500/swinunet', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Fullsup_SwinUnet_25/swinunet', iternum=10000)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Fullsup_SwinUnet_50/swinunet', iternum=20000)

    # CPS (UNeXt + Swin-unet) + contrastive loss + ema teacher
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_ema_25/Unext', iternum=10000)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_ema_50/Unext', iternum=20000)


    ################################################################################################
    # ### BUSI

    # proj_name = 'BUSI'
    # root_dir = '/mnt/storage/fangyijie/BUSI'
    # root_data = '/mnt/storage/fangyijie/BUSI/'

    # full supervision models
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/BUSI/Fullsup_Unext_16/unext', iternum=None, ssl=False)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/BUSI/Fullsup_Unext_32/unext', iternum=None, ssl=False)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/BUSI/Fullsup_Unext_324/unext', iternum=None, ssl=False)

    # loaded_model = test_model.load(f'{root_dir}/ssl/model/BUSI/Fullsup_SwinUnet_16/swinunet', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/BUSI/Fullsup_SwinUnet_32/swinunet', iternum=None)


    ################################################################################################
    # ### F_Abd

    # proj_name = 'F_Abd'
    # root_dir = '/mnt/storage/fangyijie/F_Abd'
    # root_data = '/mnt/storage/fangyijie/F_Abd/'

    # SOTA
    # MT 29, 93
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/F_Abd/MT_Unext_29/Unext', iternum=11600)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/F_Abd/MT_Unext_93/Unext', iternum=None)

    # CPS 29, 93
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/F_Abd/CPS_Unext_29/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/F_Abd/CPS_Unext_93/Unext', iternum=37200)

    # ICT 29, 93
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/F_Abd/ICT_Unext_29/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/F_Abd/ICT_Unext_93/Unext', iternum=None)

    # DAN 29, 93
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/F_Abd/DAN_Unext_29/Unext', iternum=11600)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/F_Abd/DAN_Unext_93/Unext', iternum=None)

    # UAMT 29, 93
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/F_Abd/UAMT_Unext_29/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/F_Abd/UAMT_Unext_93/Unext', iternum=None)

    # DCT 29, 93
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/F_Abd/DCT_Unext_29/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/F_Abd/DCT_Unext_93/Unext', iternum=None)

    # CT-CT 29, 93
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/F_Abd/CTCT_Unext_29/Unext', iternum=11600)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/F_Abd/CTCT_Unext_93/Unext', iternum=37200)

    # PCPCS 29, 93
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/F_Abd/PCPCS_Unext_29/Unext', iternum=13600)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/F_Abd/PCPCS_Unext_93/Unext', iternum=37200)

    # Ours 29, 93
    # 77b1166e
    # 046ed03e
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/F_Abd/Semi_Swin_Unext_Consist_proj_v2_29/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/F_Abd/Semi_Swin_Unext_Consist_proj_v2_93/Unext', iternum=37200)

    # full supervision models
    # UNeXt
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/F_Abd/Fullsup_Unext_29/unext', iternum=11600, ssl=False)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/F_Abd/Fullsup_Unext_93/unext', iternum=37200, ssl=False)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/F_Abd/Fullsup_Unext_1084/unext', iternum=None, ssl=False)

    # Swin-Unet
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/F_Abd/Fullsup_SwinUnet_29/swinunet', iternum=None, ssl=False)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/F_Abd/Fullsup_SwinUnet_93/swinunet', iternum=None, ssl=False)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/F_Abd/Fullsup_SwinUnet_1084/swinunet', iternum=None, ssl=False)    

    ################################################################################################
    # ### PSFHS

    # proj_name = 'PSFHS'
    # root_dir = '/mnt/storage/fangyijie/PSFHS'
    # root_data = '/mnt/storage/fangyijie/PSFHS/'

    # full supervision models
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/Fullsup_Unext_679/unext', iternum=38800)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/Fullsup_Unext_34/unext', iternum=13600)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/Fullsup_Unext_68/unext', iternum=27200)

    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/Fullsup_SwinUnet_34/swinunet', iternum=13600)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/Fullsup_SwinUnet_68/swinunet', iternum=27200)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/Fullsup_SwinUnet_679/swinunet', iternum=38800)


    # SOTA
    # MT 34, 68
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/MT_Unext_34/Unext', iternum=13600)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/MT_Unext_68/Unext', iternum=27200)

    # CPS 34, 68
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/CPS_Unext_34/Unext', iternum=13600)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/CPS_Unext_34/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/CPS_Unext_68/Unext', iternum=27200)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/CPS_Unext_68/Unext', iternum=None)

    # ICT 34, 68
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/ICT_Unext_34/Unext', iternum=13600)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/ICT_Unext_68/Unext', iternum=27200)

    # DAN 34, 68
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/DAN_Unext_34/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/DAN_Unext_68/Unext', iternum=None)

    # UAMT 34, 68
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/UAMT_Unext_34/Unext', iternum=13600)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/UAMT_Unext_68/Unext', iternum=27200)

    # DCT 34, 68
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/DCT_Unext_34/Unext', iternum=13600)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/DCT_Unext_68/Unext', iternum=27200)

    # CT-CT 34, 68
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/CTCT_Unext_34/Unext/old', iternum=12002)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/CTCT_Unext_34/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/CTCT_Unext_68/Unext', iternum=27200)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/CTCT_Unext_68/Unext', iternum=None)

    # PCPCS 34, 68
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/PCPCS_Unext_34/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/PCPCS_Unext_34/Unext/old', iternum=13600)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/PCPCS_Unext_68/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/PCPCS_Unext_68/Unext', iternum=27200)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/PCPCS_Unext_68/Unext', '/12_07', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/PCPCS_Unext_68/Unext', '/12_07', iternum=27200)

    # DSTCT 34, 68
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/DSTCT_Unext_34/Unext/old', iternum=13600, model_name='model1_iter_8568_dice_0.9645.pth')
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/DSTCT_Unext_68/Unext', iternum=27200)

    # Ours
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/Semi_Swin_Unext_Consist_proj_34/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/Semi_Swin_Unext_Consist_proj_34/Unext', iternum=13600)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/Semi_Swin_Unext_Consist_proj_34/Unext', iternum=15300)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/Semi_Swin_Unext_Consist_proj_34/Unext', '/15300_proj_128', iternum=15300)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/Semi_Swin_Unext_Consist_proj_68/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/Semi_Swin_Unext_Consist_proj_68/Unext', iternum=27200)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/Semi_Swin_Unext_Consist_proj_68/Unext', iternum=30600)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/Semi_Swin_Unext_Consist_proj_68/Unext', '/proj_128', iternum=30600)

    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/Semi_Swin_Unext_Consist_proj_v2_34/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/Semi_Swin_Unext_Consist_proj_v2_68/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/Semi_Swin_Unext_Consist_proj_v3_34/Unext', iternum=None)

    # Ablation Study
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/Semi_Swin_Unext_Consist_proj_v2_Ablation_68/Unext/1.0_10.0_0.5_baseline_CPS_Lmac', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/PSFHS/Semi_Swin_Unext_Consist_proj_v2_Ablation_68/Unext', iternum=None)

    ################################################################################################
    # ### Run Test

    logging.basicConfig(filename=f"{root_dir}/ssl/model/{proj_name}/test_log.txt", level=logging.INFO, 
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    test_list = 'ssl/test/data.list'

    with open(root_data + test_list, "r") as f1:
        sample_list = f1.readlines()

    dsc = DSC()
    hd = HD()

    tr_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    dsc_list = []
    dsc1_list = []
    dsc2_list = []
    iou = []
    iou1_list = []
    iou2_list = []
    hd_list = []
    hd1_list = []
    hd2_list = []
    hd95_list = []
    hd95_1_list = []
    hd95_2_list = []
    asd_list = []
    asd_1_list = []
    asd_2_list = []
    i = 0

    for idx in range(len(sample_list)):
        case = sample_list[idx].replace('\n', '')
        # print(case)
        if "F_Abd" in root_data:
            image = cv2.imread(root_data + f"image/{case}", cv2.IMREAD_COLOR) #load US images
            label = cv2.imread(root_data + f"mask/{case}".replace('_image_','_mask_'), cv2.IMREAD_UNCHANGED) #load masks
        elif "BUSI" in root_data:
            image = cv2.imread(root_data + f"image/{case}", cv2.IMREAD_COLOR) #load US images
            label = cv2.imread(root_data + f"mask/{case}".replace('.png','_mask.png'), cv2.IMREAD_GRAYSCALE) #load US images
        else:
            image = cv2.imread(root_data + f"image/{case}.png", cv2.IMREAD_COLOR) #load US images
            if "PSFHS" in root_data:
                label = cv2.imread(root_data + f"mask/{case}.png", cv2.IMREAD_UNCHANGED) #load masks
            else:
                label = cv2.imread(root_data + f"mask/{case}_Annotation.png", cv2.IMREAD_UNCHANGED) #load masks

        # if "PSFHS" not in root_data:
        #     label = cv2.imread(root_data + f"mask/{case}_Annotation.png", cv2.IMREAD_UNCHANGED) #load masks
        # else:
        #     label = cv2.imread(root_data + f"mask/{case}.png", cv2.IMREAD_UNCHANGED) #load masks

        # label = label/255
        if "PSFHS" not in root_data:
            label[label != 0] = 1.0
        label = label.astype(np.uint8)
        label = cv2.resize(label, (448,448), 0, 0, interpolation = cv2.INTER_NEAREST)
        
        pred = loaded_model.predict(image)

        # print('*'*20)
        # print(image.shape)
        # print(f'image max: {image.max()}')
        # print(f'image min: {image.min()}')
        # print(label.shape)
        # print(f'label max: {label.max()}')
        # print(f'label min: {label.min()}')
        # print(pred.shape)
        # print(f'pred max: {pred.max()}')
        # print(f'pred min: {pred.min()}')
        # print('*'*20)

        torch_label = torch.from_numpy(label)
        torch_pred = torch.from_numpy(pred)

        if "PSFHS" in root_data:
            tmp_torch_pred = F.one_hot(torch_pred.long(), 3)  # (BS,336,544,3)
            tmp_torch_label = F.one_hot(torch_label.long(), 3)  # (BS,336,544,3)    
            # print(f'tmp_torch_pred shape: {tmp_torch_pred.shape}, tmp_torch_label shape: {tmp_torch_label.shape}') 
            tmp_pred_ps = tmp_torch_pred[:, :, 1].numpy()
            tmp_label_ps = tmp_torch_label[:, :, 1].numpy()
            tmp_pred_fh = tmp_torch_pred[:, :, 2].numpy()
            tmp_label_fh = tmp_torch_label[:, :, 2].numpy()

        if "PSFHS" in root_data:
            tmp_dsc_score1, tmp_dsc_score2, tmp_avg_dsc = dsc(torch_pred, torch_label)
            # print(f'tmp_dsc_score1: {tmp_dsc_score1}, tmp_dsc_score2: {tmp_dsc_score2}, tmp_avg_dsc: {tmp_avg_dsc}')
            # print(f'tmp_avg_dsc: {tmp_avg_dsc}')
            tmp_dsc_score = tmp_avg_dsc
            dsc1_list.append(tmp_dsc_score1)
            dsc2_list.append(tmp_dsc_score2)
        else:
            tmp_dsc_score = dsc_fh(torch_pred, torch_label)

        try:
            tmp_hd_score = hd.evaluation(pred, label)
            # tmp_hd95_score = hd95(pred, label)
            # print(f'tmp HD score: {tmp_hd_score}')
            if "PSFHS" in root_data:
                tmp_hd_score_ps = hd.evaluation(tmp_pred_ps, tmp_label_ps)
                tmp_hd_score_fh = hd.evaluation(tmp_pred_fh, tmp_label_fh)
                hd1_list.append(tmp_hd_score_ps)
                hd2_list.append(tmp_hd_score_fh)

                tmp_hd95_score_ps = hd95(tmp_pred_ps, tmp_label_ps)
                tmp_hd95_score_fh = hd95(tmp_pred_fh, tmp_label_fh)
                hd95_1_list.append(tmp_hd95_score_ps)
                hd95_2_list.append(tmp_hd95_score_fh)

                tmp_asd_score_ps = asd(tmp_pred_ps, tmp_label_ps)
                tmp_asd_score_fh = asd(tmp_pred_fh, tmp_label_fh)
                asd_1_list.append(tmp_asd_score_ps)
                asd_2_list.append(tmp_asd_score_fh)
            else:
                # print(f'pred and label : {pred.shape} ... {label.shape}')
                tmp_hd95_score = hd95(pred, label)
                hd95_list.append(tmp_hd95_score)

                tmp_asd_score_fh = asd(pred, label)
                asd_list.append(tmp_asd_score_fh)
        except Exception as e:
            print(f'case: {case}')
            print(f'label unique: {np.unique(label)}')
            print(f'pred unique: {np.unique(pred)}')
            print(f'error: {e}')
            continue

        if "PSFHS" in root_data:
            tmp_iou_ps = calculate_metric_percase(tmp_pred_ps, tmp_label_ps)
            tmp_iou_fh = calculate_metric_percase(tmp_pred_fh, tmp_label_fh)
            iou1_list.append(tmp_iou_ps)
            iou2_list.append(tmp_iou_fh)

        tmp_iou_idx = calculate_metric_percase(pred, label)
        # print(f'time: {str(datetime.datetime.now())} | name: {case} | idx: {idx} | dice score: {tmp_dsc_score} | iou: {tmp_iou_idx} | hd: {tmp_hd_score}')
        # print(f'idx: {idx} | dice score: {tmp_dsc_score} | iou: {tmp_iou_idx} | hd: {tmp_hd_score}')

        logging.info(f'time: {str(datetime.datetime.now())} | name: {case} | idx: {idx} | dice score: {tmp_dsc_score} | iou: {tmp_iou_idx} | hd: {tmp_hd_score}')

        dsc_list.append(tmp_dsc_score)
        iou.append(tmp_iou_idx)
        hd_list.append(tmp_hd_score) 
        # hd95_list.append(tmp_hd95_score)

        # i += 1
        
        # if i > 3:
        #     break
        # break
    # print(f'time: {str(datetime.datetime.now())} | hd95 length: {len(hd95_list)} | asd length: {len(asd_list)} | DSC length: {len(dsc_list)}')
    # print(f'Achieve average DSC {np.mean(dsc_list)*100} on {len(dsc_list)} test images')
    # print(f'Achieve average IoU {np.mean(iou)*100} on {len(iou)} test images')
    # print(f'Achieve average HD score {np.mean(hd_list)} on {len(hd_list)} test images')
    # print(f'Achieve average HD95 score {np.mean(hd95_list)} on {len(hd95_list)} test images')
    # print(f'Achieve average ASD score {np.mean(asd_list)} on {len(asd_list)} test images')

    print(f'='*50)
    # logging.info(f'model path: {test_model.model_path}')

    logging.info(f'Achieve average DSC {np.mean(dsc_list)*100} on {len(dsc_list)} test images')
    logging.info(f'Achieve average IoU {np.mean(iou)*100} on {len(iou)} test images')
    logging.info(f'Achieve average HD score {np.mean(hd_list)} on {len(hd_list)} test images')
    logging.info(f'Achieve average HD95 score {np.mean(hd95_list)} on {len(hd95_list)} test images')
    logging.info(f'Achieve average ASD score {np.mean(asd_list)} on {len(asd_list)} test images')


    if "PSFHS" in root_data:
        # print(f'Achieve average DSC 1 {np.mean(dsc1_list)*100} on {len(dsc1_list)} test images')
        # print(f'Achieve average DSC 2 {np.mean(dsc2_list)*100} on {len(dsc2_list)} test images')
        # print(f'Achieve average IoU 1 {np.mean(iou1_list)*100} on {len(iou1_list)} test images')
        # print(f'Achieve average IoU 2 {np.mean(iou2_list)*100} on {len(iou2_list)} test images')
        # print(f'Achieve average HD95 1 score {np.mean(hd95_1_list)} on {len(hd95_1_list)} test images')
        # print(f'Achieve average HD95 2 score {np.mean(hd95_2_list)} on {len(hd95_2_list)} test images')
        # print(f'Achieve average HD 1 score {np.mean(hd1_list)} on {len(hd1_list)} test images')
        # print(f'Achieve average HD 2 score {np.mean(hd2_list)} on {len(hd2_list)} test images')
        # print(f'Achieve average ASD 1 score {np.mean(asd_1_list)} on {len(asd_1_list)} test images')
        # print(f'Achieve average ASD 2 score {np.mean(asd_2_list)} on {len(asd_2_list)} test images')

        logging.info(f'Achieve average DSC 1 {np.mean(dsc1_list)*100} on {len(dsc1_list)} test images')
        logging.info(f'Achieve average DSC 2 {np.mean(dsc2_list)*100} on {len(dsc2_list)} test images')
        logging.info(f'Achieve average IoU 1 {np.mean(iou1_list)*100} on {len(iou1_list)} test images')
        logging.info(f'Achieve average IoU 2 {np.mean(iou2_list)*100} on {len(iou2_list)} test images')
        logging.info(f'Achieve average HD95 1 score {np.mean(hd95_1_list)} with std {np.std(hd95_1_list)} on {len(hd95_1_list)} test images')
        logging.info(f'Achieve average HD95 2 score {np.mean(hd95_2_list)} with std {np.std(hd95_2_list)} on {len(hd95_2_list)} test images')
        logging.info(f'Achieve average HD 1 score {np.mean(hd1_list)} with std {np.std(hd1_list)} on {len(hd1_list)} test images')
        logging.info(f'Achieve average HD 2 score {np.mean(hd2_list)} with std {np.std(hd2_list)} on {len(hd2_list)} test images')
        logging.info(f'Achieve average ASD 1 score {np.mean(asd_1_list)} with std {np.std(asd_1_list)} on {len(asd_1_list)} test images')
        logging.info(f'Achieve average ASD 2 score {np.mean(asd_2_list)} with std {np.std(asd_2_list)} on {len(asd_2_list)} test images')
        
    ################################################################################################
    # ### Visualization

    # MT, DAN, DCT, UAMT, CPS, ICT, CTCT, PCPCS, ours(LMCT)
    # ssl_method = 'MT'

    # ESTT
    # save_dir = '/mnt/storage/fangyijie/HC18/assets/ESTT/new_pngs'
    # samplecases = ['Patient01700_Plane3_1_of_1', 'Patient00759_Plane3_1_of_4', 'Patient01648_Plane3_4_of_6']
    # samplecases = ['Patient01610_Plane3_1_of_9', 'Patient01229_Plane3_1_of_4', 'Patient00881_Plane3_1_of_2','Patient01267_Plane3_3_of_4']


    # HC18
    # save_dir = '/mnt/storage/fangyijie/HC18/assets/HC18/new_pngs'
    # samplecases = ['771_HC', '021_HC', '087_HC', '138_2HC']
    # samplecases = ['021_HC','519_HC','203_HC']
    # samplecases = ['519_HC','203_HC']

    # # idx - 323, 336
    # for idx in range(len(sample_list)):
    #     case = sample_list[idx].replace('\n', '')
    #     if case in samplecases:
    #         print(case)
    #         image = cv2.imread(root_data + f"image/{case}.png", cv2.IMREAD_COLOR) #load US images
    #         label = cv2.imread(root_data + f"mask/{case}_Annotation.png", cv2.IMREAD_UNCHANGED) #load masks
    #         label[label != 0] = 1.0
    #         label = label.astype(np.uint8)
    #         label = cv2.resize(label, (448,448), 0, 0, interpolation = cv2.INTER_NEAREST)
    #         pred = loaded_model.predict(image)

    #         cv2.imwrite(f'{save_dir}/{ssl_method}_pred_{case}.png', pred * 250)
    #         if ssl_method == 'ours':
    #             cv2.imwrite(f'{save_dir}/{ssl_method}_label_{case}.png', label * 250)
    #         # break

    # tensor_transforms = transforms.Compose([
    #     transforms.ToTensor(),
    # ])

    # torch_img = tensor_transforms(image)

    # noise = torch.clamp(torch.randn_like(torch_img) * 0.1, -0.2, 0.2)
    # noisy_img = torch_img + noise

    # noisy_img_array = noisy_img.numpy().transpose(1,2,0)

    # noisy_img_array.shape

    # cv2.imwrite(f'noisylabel_{case}.png', noisy_img_array[...,1] * 250)

    ################################################################################################
    # ### GT
    # mk_path1 = 'gt_0010.png'
    # mk_path2 = 'gt_0020.png'

    # mk1_array = cv2.imread(mk_path1, cv2.IMREAD_UNCHANGED)
    # mk2_array = cv2.imread(mk_path2, cv2.IMREAD_UNCHANGED)

    # mk1_array = mk1_array*100
    # mk2_array = mk2_array*100
    # cv2.imwrite(f"pred_{mk_path1}",mk1_array)
    # cv2.imwrite(f"pred_{mk_path2}",mk2_array)
