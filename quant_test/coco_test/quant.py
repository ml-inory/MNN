# -*- coding: utf-8 -*-
'''
随机用OVC2007的N张图片量化
'''
import os, sys
import json
import numpy as np
import time
import cv2
import onnx

def pick_images(img_num, input_folder, output_folder='./quant_images/', input_size=(320,320)):
    os.system(f'rm {output_folder}/*')
    # input_folder = 'VOCdevkit/VOC2007/JPEGImages'
    img_list = os.listdir(input_folder)
    img_list = np.random.choice(img_list, size=img_num, replace=False)
    for img_name in img_list:
        # img = cv2.imread(os.path.join(input_folder, img_name))
        # img = cv2.resize(img, input_size)
        # cv2.imwrite(os.path.join(output_folder, img_name), img)
        os.system(f'cp {os.path.join(input_folder, img_name)} {os.path.join(output_folder, img_name)}')

def quant(input_model_path, config_path, output_model, img_num=30, feature_quant='KL', weight_quant='MAX_ABS'):
    input_model = input_model_path
    config = json.load(open(config_path, 'r'))
    config['feature_quantize_method'] = feature_quant
    config['weight_quantize_method'] = weight_quant
    config['used_image_num'] = img_num
    json.dump(config, open(config_path, 'w'), indent=2)
    time.sleep(3)
    os.system(f'../../build/quantized.out {input_model} {output_model} {config_path}')

if __name__ == '__main__':
    input_folder = './train'
    img_folder = './quant_images'
    if not os.path.exists(img_folder):
        os.mkdir(img_folder)

    # 特征量化方法: KL ADMM
    # 权值量化方法: MAX_ABS ADMM
    feature_quant = 'KL'
    weight_quant = 'MAX_ABS'
    img_num_range = [30, 100, 500, 1000] if feature_quant == 'KL' else [30, 100]
    
    for img_num in img_num_range:
        pick_images(img_num, input_folder, img_folder)
        quant('./ssd_mbv2_gw.mnn', './ssd_mbv2_gw.json', './ssd_mbv2_gw_quant_%d.mnn' % img_num, img_num, feature_quant, weight_quant)
