# -*- coding: utf-8 -*-
'''
随机用OVC2007的N张图片量化
'''
import os, sys
import json
import numpy as np

def pick_images(img_num, output_folder='./quant_images/'):
    os.system(f'rm {output_folder}/*')
    input_folder = 'VOCdevkit/VOC2007/JPEGImages'
    img_list = os.listdir(input_folder)
    img_list = np.random.choice(img_list, img_num)
    for img_name in img_list:
        os.system(f'cp {os.path.join(input_folder, img_name)} {os.path.join(output_folder, img_name)}')

def quant(input_model_path, config_path):
    input_model = input_model_path
    output_model = os.path.join('models/quant/', os.path.basename(input_model))
    os.system(f'../build/quantized.out {input_model} {output_model} {config_path}')

if __name__ == '__main__':
    img_folder = './quant_images'
    if not os.path.exists(img_folder):
        os.mkdir(img_folder)

    pick_images(30, img_folder)
    quant('./models/unquant/retinanet_mbv2_voc.mnn', 'retina_quant_config.json')
    quant('./models/unquant/ssd_mbv2_voc.mnn', 'ssd_quant_config.json')
