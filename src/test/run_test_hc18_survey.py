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
import random

from torchvision import transforms
# from unet.model import model
from unext.model import model
from utils.metrics import DSC, HD, calculate_metric_percase
# from resnetunet.model import model
# from effunet.model import model
# from attunet.model import model
# from mitunet.model import model
# from swinunet.model import model as vit_seg
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

def fill_pred_with_ellipse(image_name, mask, saved_path, al_round):
    mask_uint8 = (mask * 255).astype(np.uint8)
    # image = cv2.imread(f"{image_name}.png")  # or whatever your original image is
    image = cv2.imread(root_data + f"image/{image_name}.png") #load US images        

    # 1. Find contours of the positive region
    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,       # only outer contour
        cv2.CHAIN_APPROX_NONE
    )    
    if len(contours) == 0:
        raise ValueError("No foreground region (pixel=1) found in the mask.")    
    # If multiple regions exist, take the largest one
    cnt = max(contours, key=cv2.contourArea)        

    # 2. Fit ellipse to the contour
    #    Need at least 5 points for cv2.fitEllipse
    if len(cnt) < 5:
        raise ValueError("Not enough points to fit an ellipse.")

    ellipse = cv2.fitEllipse(cnt)   # (center, (major_axis, minor_axis), angle)

    # 3. Draw a filled ellipse into a new mask
    ellipse_mask = np.zeros_like(mask_uint8)  # same size as original mask
    cv2.ellipse(
        ellipse_mask,
        ellipse,
        color=255,     # foreground value
        thickness=2   # -1 means filled
    )

    # If you want the result back as 0/1 instead of 0/255:
    ellipse_mask_binary = (ellipse_mask > 0).astype(np.uint8)

    h, w, _ = image.shape
    ellipse_mask_binary = cv2.resize(ellipse_mask_binary, (w,h), 0, 0, interpolation = cv2.INTER_NEAREST)

    overlay = image.copy()
    overlay[ellipse_mask_binary == 1] = (0, 255, 0)  # color inside ellipse
    cv2.imwrite(f"{saved_path}al/samples/{image_name}_overlay{al_round}.png", overlay)    


if __name__ == "__main__":
    args = parser.parse_args()
    random.seed(2025)

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
    # else:
    #     test_model = vit_seg(num_classes=args.num_classes)
    ################################################################################################
    # ### ES-TT

    proj_name = 'ESTT'
    root_dir = '/mnt/storage/fangyijie/ESTT'
    root_data = '/mnt/storage/fangyijie/ESTT/'

    # Ours (UNeXt x 2)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/Semi_Unext_Unext_Consist_proj_AblationStudy_17/Unext', iternum=None)
    # Ours (UNeXt x 2) + L_semi
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/Semi_Unext_Unext_Consist_proj_AS_NoCon_17/Unext', iternum=None)

    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/Semi_Swin_Unext_Consist_proj_17/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/Semi_Swin_Unext_Consist_proj_34/Unext', iternum=None)

    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/Semi_Swin_Unext_Consist_proj_v2_17/Unext', iternum=None)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/ESTT/Semi_Swin_Unext_Consist_proj_v2_34/Unext', iternum=None)

    # loaded_model = test_model.load(f'{root_dir}/al/model/ESTT/Semi_Unext_Swin_AL_3/Unext/EntropySampler', model_name='model1_iter_6000.pth')
    # loaded_model = test_model.load(f'{root_dir}/al/model/ESTT/Semi_Unext_Swin_AL_7/Unext/HybridSampler', model_name='model1_iter_14000.pth')

    # loaded_model = test_model.load(f'{root_dir}/al/model/ESTT/Semi_Unext_Swin_AL_3/Unext', model_name='Unext_best_model1.pth')
    # loaded_model = test_model.load(f'{root_dir}/al/model/ESTT/Semi_Unext_Swin_AL_7/Unext', model_name='model1_iter_14000.pth')

    # al_round = 1
    # loaded_model = test_model.load(f'{root_dir}/al/model/ESTT/Semi_Unext_Swin_AL_7/Unext', model_name='model1_iter_2800.pth')
    # al_round = 2
    # loaded_model = test_model.load(f'{root_dir}/al/model/ESTT/Semi_Unext_Swin_AL_7/Unext/HybridSampler', model_name='model1_iter_5600.pth')
    # al_round = 3
    # loaded_model = test_model.load(f'{root_dir}/al/model/ESTT/Semi_Unext_Swin_AL_7/Unext/HybridSampler', model_name='model1_iter_8400.pth')
    al_round = 4
    loaded_model = test_model.load(f'{root_dir}/al/model/ESTT/Semi_Unext_Swin_AL_7/Unext/HybridSampler', model_name='model1_iter_11200.pth')
    # al_round = 5
    # loaded_model = test_model.load(f'{root_dir}/al/model/ESTT/Semi_Unext_Swin_AL_7/Unext/HybridSampler', model_name='Unext_best_model1.pth')

    ################################################################################################
    # ### HC18

    # proj_name = 'HC18'
    # root_dir = '/mnt/storage/fangyijie/HC18'
    # root_data = '/mnt/storage/fangyijie/HC18/'

    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Swin_Unext_Consist_proj_v2_50/Unext', iternum=None)

    # Active Learning
    # ssl models 5
    # iternum = 2000, 4000, 6000, 8000, 10000
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Unet_Unet_AL_5/unet', iternum=4000)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Unet_Unet_AL_5/unet', iternum=6000)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Unet_Unet_AL_5/unet', iternum=8000)
    # loaded_model = test_model.load(f'{root_dir}/ssl/model/HC18/Semi_Unet_Unet_AL_5/unet', iternum=10000)
    # loaded_model = test_model.load(f'{root_dir}/al/model/HC18/Semi_Unext_Swin_AL_5/Unext/HybridSampler', model_name='model1_iter_10000.pth')
    # loaded_model = test_model.load(f'{root_dir}/al/model/HC18/Semi_Unext_Swin_AL_10/Unext/HybridSampler', model_name='model1_iter_20000.pth')
    # loaded_model = test_model.load(f'{root_dir}/al/model/HC18/Semi_Unext_Swin_AL_5/Unext/HybridSampler', model_name='model1_iter_10000.pth')

    # loaded_model = test_model.load(f'{root_dir}/al/model/HC18/Semi_Unext_Swin_AL_5/Unext/EntropySampler', model_name='model1_iter_6700_dice_0.976.pth')
    # loaded_model = test_model.load(f'{root_dir}/al/model/HC18/Semi_Unext_Swin_AL_10/Unext/RandomSampler', model_name='model1_iter_20000.pth')    

    # loaded_model = test_model.load(f'{root_dir}/al/model/HC18/Semi_Unext_Swin_AL_5/Unext', model_name='Unext_best_model1.pth')
    # loaded_model = test_model.load(f'{root_dir}/al/model/HC18/Semi_Unext_Swin_AL_5/Unext', model_name='model1_iter_10000.pth')    
    # loaded_model = test_model.load(f'{root_dir}/al/model/HC18/Semi_Unext_Swin_AL_10/Unext', model_name='model1_iter_4000.pth')

    # al_round = 1
    # loaded_model = test_model.load(f'{root_dir}/al/model/HC18/Semi_Unext_Swin_AL_10/Unext', model_name='model1_iter_4000.pth')
    # al_round = 2
    # loaded_model = test_model.load(f'{root_dir}/al/model/HC18/Semi_Unext_Swin_AL_10/Unext/HybridSampler_new', model_name='model1_iter_8000.pth')
    # al_round = 3
    # loaded_model = test_model.load(f'{root_dir}/al/model/HC18/Semi_Unext_Swin_AL_10/Unext/HybridSampler_new', model_name='model1_iter_12000.pth')
    # al_round = 4
    # loaded_model = test_model.load(f'{root_dir}/al/model/HC18/Semi_Unext_Swin_AL_10/Unext/HybridSampler_new', model_name='model1_iter_16000.pth')
    # al_round = 5
    # loaded_model = test_model.load(f'{root_dir}/al/model/HC18/Semi_Unext_Swin_AL_10/Unext/HybridSampler_new', model_name='Unext_best_model1.pth')

    ################################################################################################
    # ### Run Test

    logging.basicConfig(filename=f"{root_dir}/al/model/{proj_name}/test_log.txt", level=logging.INFO, 
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
    selected_k = 30

    sample_list = random.sample(sample_list, selected_k)

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
        
        # print(image.shape)
        # image = cv2.resize(image, (448,448), 0, 0, interpolation = cv2.INTER_NEAREST)
        # label = label/255
        label[label != 0] = 1.0
        label = label.astype(np.uint8)
        label = cv2.resize(label, (448,448), 0, 0, interpolation = cv2.INTER_NEAREST)
        
        pred = loaded_model.predict(image)
        # print(pred.shape)
        # print(np.min(pred))
        # print(np.max(pred))

        # fill_pred_with_ellipse(case, pred, root_data, al_round)  # fill pred with ellipse and save overlay
        # cv2.imwrite(f"{root_data}al/samples/{case}.png", image)

        torch_label = torch.from_numpy(label)
        torch_pred = torch.from_numpy(pred)

        tmp_dsc_score = dsc_fh(torch_pred, torch_label)

        try:
            tmp_hd_score = hd.evaluation(pred, label)
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

        tmp_iou_idx = calculate_metric_percase(pred, label)

        logging.info(f'time: {str(datetime.datetime.now())} | name: {case} | idx: {idx} | dice score: {tmp_dsc_score} | iou: {tmp_iou_idx} | hd: {tmp_hd_score}')

        dsc_list.append(tmp_dsc_score)
        iou.append(tmp_iou_idx)
        hd_list.append(tmp_hd_score) 

        # break  # DEBUG

    print(f'='*50)
    # logging.info(f'model path: {test_model.model_path}')

    logging.info(f'Achieve average DSC {np.mean(dsc_list)*100} on {len(dsc_list)} test images')
    logging.info(f'Achieve average IoU {np.mean(iou)*100} on {len(iou)} test images')
    logging.info(f'Achieve average HD score {np.mean(hd_list)} on {len(hd_list)} test images')
    logging.info(f'Achieve average HD95 score {np.mean(hd95_list)} on {len(hd95_list)} test images')
    logging.info(f'Achieve average ASD score {np.mean(asd_list)} on {len(asd_list)} test images')
        
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

