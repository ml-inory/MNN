//
//  facenet.cpp
//  MNN
//
//  Created by MNN on 2018/05/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
// #define MNN_OPEN_TIME_TRACE
#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <vector>

// #include <MNN/AutoTime.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

using namespace MNN;
using namespace MNN::CV;

// std::vector<std::string> VOC_NAMES{
//     "aeroplane",
//     "bicycle",
//     "bird",
//     "boat",
//     "bottle",
//     "bus",
//     "car",
//     "cat",
//     "chair",
//     "cow",
//     "diningtable",
//     "dog",
//     "horse",
//     "motorbike",
//     "person",
//     "pottedplant",
//     "sheep",
//     "sofa",
//     "train",
//     "tvmonitor",
// };

typedef struct
{
    float prob;
    CV::Rect bbox;
    std::vector<CV::Point> ldms;
} FaceOutput;

void face_config_anchor(std::vector<CV::Rect>& anchors, const cv::Size& input_size, const std::vector<std::vector<int> >& min_sizes, const std::vector<int>& strides)
{
    anchors.clear();

    float offset = 0.5f;
    for (int i = 0; i < strides.size(); i++)
    {
        // compute feature map size
        int featmap_h = (int)ceil(input_size.height * 1.0 / strides[i]);
        int featmap_w = (int)ceil(input_size.width * 1.0 / strides[i]);
        
        const std::vector<int>& min_size = min_sizes[i];
        
        for (int r = 0; r < featmap_h; r++)
        {
            for (int c = 0; c < featmap_w; c++)
            {
                for (const int msize : min_size)
                {
                    anchors.push_back(CV::Rect::MakeXYWH((c + offset) * strides[i], (r + offset) * strides[i], msize, msize));
                }
            }
        }
    }

    // printf("anchor num: %d\n", anchors.size());
}

float iou(const FaceOutput& box1, const FaceOutput& box2)
{
    int x1 = std::max(box1.bbox.x(), box2.bbox.x());
    int y1 = std::max(box1.bbox.y(), box2.bbox.y());
    int x2 = std::min((box1.bbox.x() + box1.bbox.width()), (box2.bbox.x() + box2.bbox.width()));
    int y2 = std::min((box1.bbox.y() + box1.bbox.height()),(box2.bbox.y() + box2.bbox.height()));
    float over_w = std::max(0.0f, x2 - x1 + 1.0f);
    float over_h = std::max(0.0f, y2 - y1 + 1.0f);
    float over_area = over_w * over_h;
    float iou = over_area / (box1.bbox.width() * box1.bbox.height() + box2.bbox.width() * box2.bbox.height() - over_area);
    return iou;
}

std::vector<FaceOutput> face_nms(std::vector<FaceOutput>& p_pResVec, float p_fNMSThresh = 0.3)
{
    std::vector<FaceOutput> result;

    std::sort(p_pResVec.begin(), p_pResVec.end(), [](FaceOutput lhs, FaceOutput rhs){ return lhs.prob > rhs.prob; });
    if (p_pResVec.size() > 1000)
    {
        p_pResVec.erase(p_pResVec.begin() + 1000, p_pResVec.end());
    }

    while (p_pResVec.size() > 0)
    {
        result.push_back(p_pResVec[0]);

        for(auto it = p_pResVec.begin() + 1; it != p_pResVec.end(); )
        {
            float iou_value = iou(p_pResVec[0], *it);
            if (iou_value > p_fNMSThresh)
                it = p_pResVec.erase(it);
            else
                it++;
        }
        p_pResVec.erase(p_pResVec.begin());
    }
    return result;
}

void face_get_output(std::vector<FaceOutput>& face_output, std::vector<Tensor*> ldms, std::vector<Tensor*> scores, std::vector<Tensor*> bboxes, const std::vector<CV::Rect>& anchors, float thresh = 0.5, float nms_thresh = 0.4)
{
    face_output.clear();
    int result_num = ldms[0]->shape()[1]; // H
    float* ldm_ptr = ldms[0]->host<float>();
    float* score_ptr = scores[0]->host<float>();
    float* bbox_ptr = bboxes[0]->host<float>();

    for (int i = 0; i < result_num; i++, ldm_ptr+=10, score_ptr+=2, bbox_ptr+=4)
    {
        float prob = score_ptr[1];

        if (prob > thresh)
        {
            // bbox decode
            float bbox_x = bbox_ptr[0];
            float bbox_y = bbox_ptr[1];
            float bbox_w = bbox_ptr[2];
            float bbox_h = bbox_ptr[3];

            float anchor_w = anchors[i].width(); // * stride;
            float anchor_h = anchors[i].height();// * stride;
            float anchor_x = anchors[i].x();
            float anchor_y = anchors[i].y();

            bbox_x = 0.1 * bbox_x * anchor_w + anchor_x;
            bbox_y = 0.1 * bbox_y * anchor_h + anchor_y;
            bbox_w = expf(0.2 * bbox_w) * anchor_w;
            bbox_h = expf(0.2 * bbox_h) * anchor_h;

            if (bbox_w > 0 && bbox_h > 0)
            {
                FaceOutput result;
                result.prob = prob;
                result.bbox = CV::Rect::MakeXYWH(bbox_x - bbox_w / 2, bbox_y - bbox_h / 2, bbox_w, bbox_h);
                result.ldms.resize(5);
                for (int n = 0; n < 5; n++)
                {
                    result.ldms[n].set(anchor_x + ldm_ptr[2 * n] * 0.1 * anchor_w, anchor_y + ldm_ptr[2 * n + 1] * 0.1 * anchor_h);
                }
                face_output.push_back(result);
            }
        }
    }

    // printf("nms %d\n", p_stResultVec.size());
    face_output = face_nms(face_output, nms_thresh);
}

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        MNN_PRINT("Usage: ./facenet.out model.mnn input.jpg [word.txt]\n");
        return 0;
    }
    std::shared_ptr<Interpreter> net(Interpreter::createFromFile(argv[1]));
    ScheduleConfig config;
    config.type  = MNN_FORWARD_OPENCL;
    // BackendConfig bnconfig;
    // bnconfig.precision = BackendConfig::Precision_Low;
    // config.backendConfig = &bnconfig;
    auto session = net->createSession(config);

    auto input = net->getSessionInput(session, NULL);
    auto shape = input->shape();
    shape[0]   = 1;
    net->resizeTensor(input, shape);
    net->resizeSession(session);
    auto outputs = net->getSessionOutputAll(session);

    printf("input size: %d %d %d %d\n", shape[0], shape[1], shape[2], shape[3]);

    // anchor
    std::vector<CV::Rect> anchors;
    cv::Size input_size(shape[3], shape[2]);
    std::vector<std::vector<int> > min_sizes{{10, 20}, {32, 64}, {128, 256}};
    std::vector<int> strides{8, 16, 32};

    face_config_anchor(anchors, input_size, min_sizes, strides);
    
    // for (auto it = outputs.begin(); it != outputs.end(); it++)
    // {
    //     Tensor* tensor = it->second;
    //     auto shape = tensor->shape();
    //     if (shape.size() == 4)
    //         printf("name: %s  shape: %d %d %d %d\n", it->first.c_str(), shape[0], shape[1], shape[2], shape[3]);
    //     else if (shape.size() == 3)
    //         printf("name: %s  shape: %d %d %d\n", it->first.c_str(), shape[0], shape[1], shape[2]);
    // }
   
    auto dims    = input->shape();
    int inputDim = 0;
    int size_w   = 0;
    int size_h   = 0;
    int bpp      = 0;
    bpp          = input->channel();
    size_h       = input->height();
    size_w       = input->width();
    if (bpp == 0)
        bpp = 1;
    if (size_h == 0)
        size_h = 1;
    if (size_w == 0)
        size_w = 1;
    // MNN_PRINT("input: w:%d , h:%d, bpp: %d\n", size_w, size_h, bpp);

    // auto inputPatch = argv[2];
    int width, height, channel;
    
    // auto inputImage = stbi_load(inputPatch, &width, &height, &channel, 4);
    // if (nullptr == inputImage) {
    //     MNN_ERROR("Can't open %s\n", inputPatch);
    //     return 0;
    // }
    // MNN_PRINT("origin size: %d, %d\n", width, height);
    Matrix trans;
    // Set transform, from dst scale to src, the ways below are both ok
#ifdef USE_MAP_POINT
    float srcPoints[] = {
        0.0f, 0.0f,
        0.0f, (float)(height-1),
        (float)(width-1), 0.0f,
        (float)(width-1), (float)(height-1),
    };
    float dstPoints[] = {
        0.0f, 0.0f,
        0.0f, (float)(size_h-1),
        (float)(size_w-1), 0.0f,
        (float)(size_w-1), (float)(size_h-1),
    };
    trans.setPolyToPoly((Point*)dstPoints, (Point*)srcPoints, 4);
#else
    // trans.setScale((float)(width-1) / (size_w-1), (float)(height-1) / (size_h-1));
#endif
    ImageProcess::Config pre_config;
    pre_config.filterType = BILINEAR;
    // mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
    float mean[3]     = {123, 104, 117};
    float normals[3] = {1.f, 1.f, 1.f};
    // float mean[3]     = {127.5f, 127.5f, 127.5f};
    // float normals[3] = {0.00785f, 0.00785f, 0.00785f};
    ::memcpy(pre_config.mean, mean, sizeof(mean));
    ::memcpy(pre_config.normal, normals, sizeof(normals));
    pre_config.sourceFormat = BGR;
    pre_config.destFormat   = RGB;

    std::shared_ptr<ImageProcess> pretreat(ImageProcess::create(pre_config));
    // pretreat->setMatrix(trans);
    // pretreat->convert((uint8_t*)inputImage, width, height, 0, input);
    // stbi_image_free(inputImage);

    cv::Mat img = cv::imread(argv[2]);

    width = img.cols;
    height = img.rows;
    channel = img.channels();

    float w_scale = width * 1.0f / size_w;
    float h_scale = height * 1.0f / size_h;

    trans.setScale((float)(width-1) / (size_w-1), (float)(height-1) / (size_h-1));
    pretreat->setMatrix(trans);
    pretreat->convert((uint8_t*)img.data, width, height, 0, input);

    net->runSession(session);
    // 将输出分成score 和 bbox
    outputs = net->getSessionOutputAll(session);
    
    Tensor* src_ldms, *src_scores, *src_bboxes;
    Tensor* dst_ldms, *dst_scores, *dst_bboxes;
    
    auto output_it = outputs.begin();
    src_ldms = output_it->second; output_it++;
    src_scores = output_it->second; output_it++;
    src_bboxes = output_it->second;

    dst_ldms = new Tensor(src_ldms, Tensor::CAFFE);
    src_ldms->copyToHostTensor(dst_ldms);

    dst_scores = new Tensor(src_scores, Tensor::CAFFE);
    src_scores->copyToHostTensor(dst_scores);

    dst_bboxes = new Tensor(src_bboxes, Tensor::CAFFE);
    src_bboxes->copyToHostTensor(dst_bboxes);

    std::vector<FaceOutput> face_output;
    std::vector<Tensor*> ldms{dst_ldms}, scores{dst_scores}, bboxes{dst_bboxes};
    face_get_output(face_output, ldms, scores, bboxes, anchors, 0.9, 0.3);

    printf("detected %d faces\n", face_output.size());
    for (const auto& out : face_output)
    {
        // printf("cls: %d  prob: %f\n", out.cls, out.prob);
        // char text[64];
        // sprintf(text, "%s: %.2f", VOC_NAMES[out.cls].c_str(), out.prob);
        // cv::putText(img, text, cv::Point(out.bbox.x() * w_scale, out.bbox.y()* h_scale - 5), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0,0,255),1);
        cv::Rect draw_bbox = cv::Rect(out.bbox.x() * w_scale, out.bbox.y() * h_scale, out.bbox.width() * w_scale, out.bbox.height() * h_scale);
        cv::rectangle(img, draw_bbox, cv::Scalar(0,255,0), 2);
        for (const auto& pt : out.ldms)
        {
            cv::circle(img, cv::Point(pt.fX * w_scale, pt.fY * h_scale), 1, cv::Scalar(0,255,0), -1); 
        }
    }

    cv::imwrite("face_result.jpg", img);

    delete dst_ldms;
    delete dst_scores;
    delete dst_bboxes;

    return 0;
}
