# Demo - train the style transfer network & use it to generate an image

from __future__ import print_function

import time

from train_recons import train_recons
from generate import generate
from utils import list_images
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# IS_TRAINING = True
IS_TRAINING = False

BATCH_SIZE = 2
EPOCHES = 4

SSIM_WEIGHTS = [1, 10, 100, 1000]
MODEL_SAVE_PATHS = [
    'D:/project/GitHub/ImageFusion/imagefusion_deep_dense_block/models/deepfuse_dense_model_bs2_epoch4_all_weight_1e0.ckpt',
    'D:/project/GitHub/ImageFusion/imagefusion_deep_dense_block/models/deepfuse_dense_model_bs2_epoch4_all_weight_1e1.ckpt',
    'D:/project/GitHub/ImageFusion/imagefusion_deep_dense_block/models/deepfuse_dense_model_bs2_epoch4_all_weight_1e2.ckpt',
    'D:/project/GitHub/ImageFusion/imagefusion_deep_dense_block/models/deepfuse_dense_model_bs2_epoch4_all_weight_1e3.ckpt',
]

# MODEL_SAVE_PATH = './models/deepfuse_dense_model_bs4_epoch2_relu_pLoss_noconv_test.ckpt'

# model_pre_path  = './models/deepfuse_dense_model_bs2_epoch2_relu_pLoss_noconv_NEW.ckpt'
model_pre_path  = None

def main():

    if IS_TRAINING:

        original_imgs_path = list_images('D:/ImageDatabase/Image_fusion_MSCOCO/original/')

        for ssim_weight, model_save_path in zip(SSIM_WEIGHTS, MODEL_SAVE_PATHS):
            print('\nBegin to train the network ...\n')
            train_recons(original_imgs_path, model_save_path, model_pre_path, ssim_weight, EPOCHES, BATCH_SIZE, debug=True)

            print('\nSuccessfully! Done training...\n')
    else:

        # sourceA_name = 'VIS'
        # sourceB_name = 'IR'
        # print('\nBegin to generate pictures ...\n')
        #
        # content_name = 'images/IV_images/' + sourceA_name
        # style_name   = 'images/IV_images/' + sourceB_name

        sourceA_name = 'image'
        sourceB_name = 'image'
        print('\nBegin to generate pictures ...\n')

        content_name = 'images/multifocus_images/' + sourceA_name
        style_name = 'images/multifocus_images/' + sourceB_name

        # fusion_type = 'addition'
        fusion_type = 'l1'
        # fusion_type = 'weight' # Failed
        for ssim_weight, model_save_path in zip(SSIM_WEIGHTS, MODEL_SAVE_PATHS):
            output_save_path = 'outputs/fused_deepdense_bs2_epoch4_all_l1_focus_'+str(ssim_weight)
            for i in range(20):
                index = i + 1
                content_path = content_name + str(index) + '_left.png'
                style_path = style_name + str(index) + '_right.png'
                generate(content_path, style_path, model_save_path, model_pre_path, ssim_weight, index, fusion_type, output_path=output_save_path)

        # print('\ntype(generated_images):', type(generated_images))
        # print('\nlen(generated_images):', len(generated_images), '\n')


if __name__ == '__main__':
    main()

