# -*- coding: utf-8 -*-
import numpy as np
import MNN
import cv2
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
from tempfile import NamedTemporaryFile

np.set_printoptions(precision=4, suppress=True)

__all__ = ['init_net', 'run_net']

def preprocess(img, input_size=(300, 300), mean=(123.675, 116.28, 103.53), std=(1, 1, 1)):
    img = cv2.resize(img, input_size).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.subtract(img, np.array(mean, dtype=np.float64).reshape(1, -1))
    img = cv2.multiply(img, np.array(std, dtype=np.float64).reshape(1, -1))
    img = img.transpose((2, 0, 1))
    return img

def config_anchor_retina(net, session, octave_base_scale=4, scales_per_octave=3, ratios=(0.5, 1.0, 2.0), strides=(8, 16, 32, 64, 128)):
    output_tensors = net.getSessionOutputAll(session)
    output_shapes = [x.getShape() for x in output_tensors.values()]
    output_shapes.sort(key=lambda x: (x[2], x[1]), reverse=True)
    output_shapes = output_shapes[::2]
    
    scales = []
    for i in range(scales_per_octave):
        scales.append(octave_base_scale * pow(2, i * 1.0 / scales_per_octave))

    anchors = []
    for i, stride in enumerate(strides):
        branch_anchor = []
        for ratio in ratios:
            h_ratio = np.sqrt(ratio)
            w_ratio = 1.0 / h_ratio
            for scale in scales:
                branch_anchor.append((stride * 0.5, stride * 0.5, stride * scale * w_ratio, stride * scale * h_ratio))
        branch_anchor = np.array(branch_anchor).flatten()
        _, _, h, w = output_shapes[i]
        grid = np.zeros((1, branch_anchor.shape[0], h, w), dtype=np.float32)
        # x grid
        grid[0, 0::4, ...] = np.tile(stride * np.arange(w), (h, 1))
        # y grid
        grid[0, 1::4, ...] = np.tile(stride * np.arange(h), (w, 1)).T
        grid[0, :, ...] += branch_anchor[:, None, None]

        # print(grid[0, 12:16, 2, 2])
        anchors.append(grid)
    return anchors

def config_anchor_ssd(net, session, scale_major=False, input_size=300, basesize_ratio_range=(0.2, 0.95), strides=[16, 30, 60, 100, 150, 300], ratios=[[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]):
    output_tensors = net.getSessionOutputAll(session)
    output_shapes = [x.getShape() for x in output_tensors.values()]
    output_shapes.sort(key=lambda x: (x[2], x[1]), reverse=True)
    output_shapes = output_shapes[::2]
    
    min_ratio, max_ratio = basesize_ratio_range
    min_ratio = int(100 * min_ratio)
    max_ratio = int(100 * max_ratio)

    num_levels = len(strides)
    step = int(np.floor(max_ratio - min_ratio) / (num_levels - 2))
    min_sizes = []
    max_sizes = []
    #VOC
    min_sizes.append(int(input_size * 10 / 100))
    max_sizes.append(int(input_size * 20 / 100))
    for r in range(int(min_ratio), int(max_ratio), step):
        min_sizes.append(int(input_size * r / 100))
        max_sizes.append(int(input_size * (r + step) / 100))
    
    anchors = []
    for k, stride in enumerate(strides):
        anchor = []

        offset = stride * 0.5
        scales = [1.0, np.sqrt(max_sizes[k] * 1.0 / min_sizes[k])]
        base_size = min_sizes[k]

        anchor_ratio = [1.0]
        for r in ratios[k]:
            anchor_ratio.append(1.0 / r)
            anchor_ratio.append(r)

        tmp_anchor = []
        if scale_major:
            for i in range(len(anchor_ratio)):
                h_ratio = np.sqrt(anchor_ratio[i])
                w_ratio = 1.0 / h_ratio
                for j in range(2):
                    anchor.append([offset, offset, base_size * w_ratio * scales[j], base_size * h_ratio * scales[j]])
                break
        else:
            for i in range(2):
                for j in range(len(anchor_ratio)):
                    h_ratio = np.sqrt(anchor_ratio[j])
                    w_ratio = 1.0 / h_ratio

                    anchor.append([offset, offset, base_size * w_ratio * scales[i], base_size * h_ratio * scales[i]])
                h_ratio = np.sqrt(anchor_ratio[0])
                w_ratio = 1.0 / h_ratio
                tmp_anchor = [offset, offset, base_size * w_ratio * scales[i+1], base_size * h_ratio * scales[i+1]]
                break
        anchor.insert(1, tmp_anchor)
        branch_anchor = np.array(anchor).flatten()
        _, _, h, w = output_shapes[k]
        grid = np.zeros((1, branch_anchor.shape[0], h, w), dtype=np.float32)
        # x grid
        grid[0, 0::4, ...] = np.tile(stride * np.arange(w), (h, 1))
        # y grid
        grid[0, 1::4, ...] = np.tile(stride * np.arange(h), (w, 1)).T
        grid[0, :, ...] += branch_anchor[:, None, None]

        # print(grid[0, 12:16, 2, 2])
        anchors.append(grid)
    return anchors

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.minimum(np.maximum(x, -20.0), 20.0)))

def softmax(x, cls_num):
    c, h, w = x.shape[1:]
    anchor_num = c // cls_num
    x = np.exp(x)
    for i in range(anchor_num):
        for row in range(h):
            for col in range(w):
                x[:, i * cls_num : (i+1)*cls_num, row, col] /= np.sum(x[:, i * cls_num : (i+1)*cls_num, row, col])
    return x

def py_cpu_nms(dets, cls_num, thresh=0.3):
    classes = dets[:,0]
    keep = []
    for c in range(cls_num - 1):
        clas_ind = np.where(classes == c)[0]
        clas = dets[clas_ind,0]
        x1 = dets[clas_ind,2]
        y1 = dets[clas_ind,3]
        x2 = x1 + dets[clas_ind,4]
        y2 = y1 + dets[clas_ind,5]
        areas = (y2-y1+1) * (x2-x1+1)
        scores = dets[clas_ind,1]
    
        index = scores.argsort()[::-1]
        while index.size >0:
            i = index[0]       # every time the first is the biggst, and add it directly
            keep.append([clas[i], scores[i], x1[i], y1[i], x2[i] - x1[i], y2[i] - y1[i]])
    
            x11 = np.maximum(x1[i], x1[index[1:]])    # calculate the points of overlap 
            y11 = np.maximum(y1[i], y1[index[1:]])
            x22 = np.minimum(x2[i], x2[index[1:]])
            y22 = np.minimum(y2[i], y2[index[1:]])

            w = np.maximum(0, x22-x11+1)    
            h = np.maximum(0, y22-y11+1)    
        
            overlaps = w*h
            
            ious = overlaps / (areas[i]+areas[index[1:]] - overlaps)
            idx = np.where(ious<=thresh)[0]
            index = index[idx+1]   # because index start from 1
    keep = np.array(keep)
    if keep.shape[0] > 200:
        # print('Too much')
        ind = keep[:, 1].argsort()[::-1]
        keep = keep[ind[:200]]
    return keep.tolist()

def postprocess(output_datas, anchors, bbox_scale, thresh=0.5, nms_thresh=0.45, target_stds=[0.1, 0.1, 0.2, 0.2]):
    output_num = len(output_datas) // 2
    anchor_num = anchors[0].shape[1] // 4
    cls_num = output_datas[0].shape[1] // anchor_num

    prob = output_datas[::2]
    bbox = output_datas[1::2]
    result = np.zeros((0, 6), dtype=np.float32)
    # sigmoid prob
    for i in range(output_num):
        prob_i = softmax(prob[i], cls_num)
        bbox_i = bbox[i]
        anchor = anchors[i]
        h, w = prob_i.shape[2:]
        clas_i = np.zeros((1, anchor_num, h, w), dtype=np.int32)
        max_prob_i = np.zeros((1, anchor_num, h, w), dtype=np.float32)
        
        for k in range(anchor_num):
            clas_i[:, k, ...] = np.argmax(prob_i[:, k*cls_num:(k+1)*cls_num - 1, ...], axis=1)
            max_prob_i[:, k, ...] = np.max(prob_i[:, k*cls_num:(k+1)*cls_num - 1, ...], axis=1)

        bbox_i[:, ::4, ...] = (target_stds[0] * bbox_i[:, ::4, ...] * anchor[:, 2::4, ...] + anchor[:, ::4, ...]) * bbox_scale[0]
        bbox_i[:, 1::4, ...] = (target_stds[1] * bbox_i[:, 1::4, ...] * anchor[:, 3::4, ...] + anchor[:, 1::4, ...]) * bbox_scale[1]
        bbox_i[:, 2::4, ...] = np.exp(target_stds[2] * bbox_i[:, 2::4, ...]) * anchor[:, 2::4, ...] * bbox_scale[0]
        bbox_i[:, 3::4, ...] = np.exp(target_stds[3] * bbox_i[:, 3::4, ...]) * anchor[:, 3::4, ...] * bbox_scale[1]

        bbox_i[:, ::4, ...] -= bbox_i[:, 2::4, ...] / 2.0
        bbox_i[:, 1::4, ...] -= bbox_i[:, 3::4, ...] / 2.0

        bbox_i[:, ::4, ...] = np.maximum(0, bbox_i[:, ::4, ...])
        bbox_i[:, 1::4, ...] = np.maximum(0, bbox_i[:, 1::4, ...])

        flat_clas = clas_i.transpose((0, 2, 3, 1)).reshape(-1, 1)
        flat_prob = max_prob_i.transpose((0, 2, 3, 1)).reshape(-1, 1)
        flat_bbox = bbox_i.transpose((0, 2, 3, 1)).reshape(-1, 4)

        ind = np.where(flat_prob > thresh)[0]
        flat_clas = flat_clas[ind]
        flat_prob = flat_prob[ind]
        flat_bbox = flat_bbox[ind]

        result = np.concatenate((result, np.hstack([flat_clas, flat_prob, flat_bbox])))

    keep = py_cpu_nms(result, cls_num, nms_thresh)
    return keep

def init_net(model_path):
    net = MNN.Interpreter(model_path)
    session = net.createSession({'numThread': 8})
    anchors = config_anchor_ssd(net, session)
    return (net, session, anchors)

def run_net(net, session, img, anchors, thresh=0.02, nms_thresh=0.45, mean=(123.675, 116.28, 103.53), std=(1,1,1)):
    input_tensor = net.getSessionInput(session)
    N, C, H, W = input_tensor.getShape()
    bbox_scale = (img.shape[1] * 1.0 / W, img.shape[0] * 1.0 / H)
    input_data = preprocess(img, (W, H), mean, std)
    tmp_input = MNN.Tensor((N, C, H, W), MNN.Halide_Type_Float,\
                    input_data, MNN.Tensor_DimensionType_Caffe)
    input_tensor.copyFromHostTensor(tmp_input)
    net.runSession(session)
    output_tensors = net.getSessionOutputAll(session)
    output_datas = []
    for output_tensor in output_tensors.values():
        output_shape = output_tensor.getShape()
        tmp_output = MNN.Tensor(output_shape, MNN.Halide_Type_Float, np.zeros(output_shape).astype(np.float32), MNN.Tensor_DimensionType_Caffe)
        output_tensor.copyToHostTensor(tmp_output)
        output_datas.append(tmp_output.getData().copy())
    output_datas.sort(key=lambda x: (x.shape[2], x.shape[1]), reverse=True)
    result = postprocess(output_datas, anchors, bbox_scale, thresh, nms_thresh)

    return result

def write_result(det_file, det_result):
    detection_res = []
    for img_id, clas, prob, x, y, w, h in det_result:
        prob = round(prob, 4)
        x = round(x, 1)
        y = round(y, 1)
        w = round(w, 1)
        h = round(h, 1)
        detection_res.append({
            'score': prob,
            'category_id': int(clas),
            'bbox': [x, y, w, h],
            'image_id': img_id
        })
    content = json.dumps(detection_res)
    with open(det_file, 'w') as f:
        f.write(content)
    
def getImgID(anno, img_name):
    for obj in anno['images']:
        if obj['file_name'] == img_name:
            return obj['id']
    return None

def save_coco(cocoEval, filename):
    with open(filename, 'w') as f:
        f.write('mAP: %f\n' % cocoEval.stats[0])
        f.write('mAP50: %f' % cocoEval.stats[1])

def __demo__():
    net, session, anchors = init_net('../models/unquant/wheat_detect.mnn')
    img = cv2.imread('../quant_images/0abc443ae.jpg')
    result = run_net(net, session, img, anchors)
    print(len(result))
    # print(result)
    for _, _, x, y, w, h in result:
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.imwrite('result.jpg', img)

def __test__():
    img_folder = './train'
    label_file = './wheat-valid.json'
    model_paths = ('./ssd_mbv2_gw.mnn', './ssd_mbv2_gw_quant_30.mnn', './ssd_mbv2_gw_quant_100.mnn', './ssd_mbv2_gw_quant_500.mnn', './ssd_mbv2_gw_quant_1000.mnn')
    thresh = 0.02
    nms_thresh = 0.45

    coco_gt = COCO(label_file)
    cat_ids = coco_gt.getCatIds(catNms=('usask_1', 'arvalis_1', 'inrae_1', 'ethz_1', 'arvalis_3', 'rres_1', 'arvalis_2'))
    img_ids = coco_gt.getImgIds()
    
    anno = json.load(open(label_file, 'r'))
    img_num = len(anno['images'])
    for model_path in model_paths:
        print('Evaluate ', model_path)
        net, session, anchors = init_net(model_path)
        det_file = os.path.basename(model_path).split('.')[0] + '_det.json'
        if os.path.exists(det_file):
            coco_dt = coco_gt.loadRes(det_file)
            cocoEval = COCOeval(coco_gt, coco_dt, 'bbox')
            cocoEval.evaluate()
            cocoEval.accumulate()
            print('COCO Eval for', model_path) 
            cocoEval.summarize()
            save_coco(cocoEval, os.path.basename(model_path).split('.')[0] + '_map.txt')
            continue
        det_result = []
        for n, obj in enumerate(anno['images']):
            img_path = os.path.join(img_folder, obj['file_name'])
            img = cv2.imread(img_path)
            if img is None:
                print('Cannot read %s' % img_path)
                continue
            result = run_net(net, session, img, anchors, thresh, nms_thresh)
            img_id = int(img_ids[n])
            for i in range(len(result)):
                result[i].insert(0, img_id)
            det_result.extend(result)
            if n % 100 == 0:
                print('%d/%d' % (n, img_num))
        write_result(det_file, det_result)
        coco_dt = coco_gt.loadRes(det_file)
        cocoEval = COCOeval(coco_gt, coco_dt, 'bbox')
        cocoEval.params.catIds = cat_ids
        cocoEval.params.imgIds = img_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        print('COCO Eval for', model_path)  
        cocoEval.summarize()
        save_coco(cocoEval, os.path.basename(model_path).split('.')[0] + '_map.txt')
        # break

if __name__ == '__main__':
    # __demo__()
    __test__()