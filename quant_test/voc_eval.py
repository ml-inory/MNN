# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np
'''
评估函数
'''
 
# 读取xml文件
def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)
 
    return objects
 
 
# 计算AP的函数
def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    计算AP值，若use_07_metric=true,则用11个点采样的方法，将rec从0-1分成11个点，这些点prec值求平均近似表示AP
    若use_07_metric=false,则采用更为精确的逐点积分方法
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
 
        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
 
        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
 
        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
 
 
def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False,
             use_diff=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    #imagesetfile，路径VOCdevkit/VOC20xx/ImageSets/Main/test.txt这里假设测试图像1000张，那么该txt文件1000行。
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file
 
    # first load gt
    # 读取真实的标签
    # if cachedir不存在就创建一个
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, '%s_annots.pkl' % imagesetfile)
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()  # 读取所有图片名
    imagenames = [x.strip() for x in lines]  # x.strip()代表去除开头和结尾的'\n'或者'\t'
 
    # 如果缓存路径对应的文件没有，则读取annotations
    if not os.path.isfile(cachefile):
        # load annotations
        # 这是一个字典
        recs = {}
        for i, imagename in enumerate(imagenames):
            # parse_rec用于读取xml文件
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)  # dump是序列化保存，load是序列化解析
    else:  # 如果已经有了cachefile缓存文件，直接读取
        # load
        with open(cachefile, 'rb') as f:
            try:
                recs = pickle.load(f)
            except:
                recs = pickle.load(f, encoding='bytes')
 
    # extract gt objects for this class
    #
    class_recs = {}  # 当前类别的标注
    npos = 0
    for imagename in imagenames:
        # recs[imagename]是保存了图片的object里面的所有属性，是个字典
        # 值保留指定类别的项
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        # 获得所有的bbox,里面保存了xmin,ymin,xmax,ymax
        bbox = np.array([x['bbox'] for x in R])
        if use_diff:  # 如果使用difficult(难检测的),所有的值都是false
            difficult = np.array([False for x in R]).astype(np.bool)
        else:  # 否则里面的内容有1有0
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        # len(R)就是当前类别的个数
        # 开辟一个全为False长度是len(R)的数组
        det = [False] * len(R)
        # 我测试~difficult的意思是取相反数之后再减1，这是什么意思。。。
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}
 
    # read dets
    # dets是检测结果的路径，读取出来就是图片名字、得分、bbox四个值。
    detfile = detpath.format(classname)
    # 读取txt里面的所有内容
    with open(detfile, 'r') as f:
        lines = f.readlines()
    # 这里的操作x.strip().split(' ')首先去掉了每行开头和结尾的'\n'或'\t'然后去除每行的' '
    # 之前再运行代码时，报错时一直以为空格会影响文件的读入，还专门写了函数去去除所有额外的空格。。多余了
    splitlines = [x.strip().split(' ') for x in lines]
    # 以图片的名称作为下标
    image_ids = [x[0] for x in splitlines]  # x[0]为名称
    confidence = np.array([float(x[1]) for x in splitlines])  # x[1]为得分
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])  # x[2]为bbox的4个值
    # 统计检测的目标数量
    nd = len(image_ids)
    # tp:正类预测为正类->A预测成A
    # fp:正类预测为负类->A预测成B
    tp = np.zeros(nd)
    fp = np.zeros(nd)
 
    if BB.shape[0] > 0:  # 行数>0
        # sort by confidence
        # 按照分数从大到小排序，返回下标
        sorted_ind = np.argsort(-confidence)
        # 按照分数从大到小排序，返回分数
        # 下面也没有用到该变量，其实sorted_ind得到之后，直接根据confidence[sorted_ind[i]]就可以得到sorted_scores
        sorted_scores = np.sort(-confidence)
        # 对BB也重排一下
        BB = BB[sorted_ind, :]
        # image_ids也重排
        image_ids = [image_ids[x] for x in sorted_ind]
        # 上面这些操作就是为了下标对应起来，后面好操作
        # go down dets and mark TPs and FPs
        for d in range(nd):
            '''
            class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}
            '''
            # 由image_ids[d]获取名称。然后得到R
            R = class_recs[image_ids[d]]
            # dets是检测结果的路径,BB通过dets获取每一行的数据，然后得到对应的BB值(也就是4个属性)
            bb = BB[d, :].astype(float)
            # 设置一个负无穷
            ovmax = -np.inf
            # BBGT是真实的坐标
            BBGT = R['bbox'].astype(float)
 
            if BBGT.size > 0:  # 如果存在GT计算交并比，如果不存在，首先就是检测错误
                # compute overlaps
                # intersection
                # 得到重叠区域，也就是左上角坐标取最大，右下角坐标取最小
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                # 计算重叠区域的面积
                inters = iw * ih
 
                # union
                # 并集面积就是两个区域的面积减去重叠区域的面积
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
                # 计算IOU，注意这个overlaps不一定是一个数值，可能是一个列表的,所有后面才有np.max和np.argmax
                overlaps = inters / uni
                # 保留最大的IOU
                ovmax = np.max(overlaps)
                # 保留最大IOU的下标
                jmax = np.argmax(overlaps)
 
            if ovmax > ovthresh:  # 这个阙值默认0.5
                if not R['difficult'][jmax]:  # 这个后面是不是少个else，如果是难测样本呢？？
                    # R = class_recs[image_ids[d]]
                    if not R['det'][jmax]:  # R['det']初始值全为False,意思应该是如果该位置第一次使用，才可以。那也会出现tp[d]=fp[d]=1的情况啊
                        # 下面都是标记
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.
 
    # compute precision recall
    # 我测试cumsum是前缀和的意思？？？？ 不懂
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
 
    return rec, prec, ap

if __name__ == '__main__':
    import sys
    classnames = [s.split('_')[0] for s in os.listdir('VOCdevkit/VOC2007/ImageSets/Main/')]
    mAP = 0.0
    for c in classnames:
        _, _, ap = voc_eval('./%s/{}.txt' % sys.argv[1],
                'VOCdevkit/VOC2007/Annotations/{}.xml',
                'VOCdevkit/VOC2007/ImageSets/Main/%s_test.txt',
                c,
                './cache')
        mAP += ap
    mAP /= len(classnames)
    print('mAP: ', mAP)