//
//  retinanet.cpp
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

std::vector<std::string> VOC_NAMES{
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
};

typedef struct
{
    int cls;
    float prob;
    CV::Rect bbox;

    // RetinaOutput& operator &= (const RetinaOutput& other)
    // {
    //     bbox.fLeft = std::max(bbox.fLeft, other.fLeft);
    //     bbox.fRight = std::max(bbox.fRight, other.fRight);
    //     bbox.
    // }
} RetinaOutput;

void retina_config_anchor(std::vector<std::vector<CV::Rect> >& anchors, int octave_base_scale, int scales_per_octave, const std::vector<float>& ratios, const std::vector<int>& strides)
{
    anchors.clear();

    std::vector<float> scales;
    for (int i = 0; i < scales_per_octave; i++)
    {
        scales.push_back(octave_base_scale * pow(2, i * 1.0f / scales_per_octave));
    }

    for (const auto& stride : strides)
    {
        std::vector<CV::Rect> branch_anchor;
        for (const auto& ratio : ratios)
        {
            float h_ratio = sqrtf(ratio);
            float w_ratio = 1.0f / h_ratio;
            for (const auto& scale : scales)
            {
                branch_anchor.push_back(CV::Rect::MakeXYWH(stride * 0.5, stride * 0.5, stride * scale * w_ratio, stride * scale * h_ratio));
            }
        }
        anchors.push_back(branch_anchor);
    }
}

void sigmoid(std::vector<Tensor*>& scores)
{
    for (auto& score : scores)
    {
        float* src = score->host<float>();
        int c, h, w;
        c = score->shape()[1];
        h = score->shape()[2];
        w = score->shape()[3];
        for (int i = 0; i < c * h * w; i++, src++)
        {
            *src = 1.0f / (1.0 + expf(-(*src)));
            // if (i < 10)
            // {
            //     printf("%f ", *src);
            // }
        }
        // printf("\n");
    }
}

float iou(const RetinaOutput& box1, const RetinaOutput& box2)
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

std::vector<RetinaOutput> retina_nms(std::vector<RetinaOutput>& p_pResVec, float p_fNMSThresh = 0.3)
{
    std::vector<RetinaOutput> result;

    std::sort(p_pResVec.begin(), p_pResVec.end(), [](RetinaOutput lhs, RetinaOutput rhs){ return lhs.prob > rhs.prob; });
    if (p_pResVec.size() > 1000)
    {
        p_pResVec.erase(p_pResVec.begin() + 1000, p_pResVec.end());
    }

    while (p_pResVec.size() > 0)
    {
        result.push_back(p_pResVec[0]);

        for(auto it = p_pResVec.begin() + 1; it != p_pResVec.end(); )
        {
            // if (p_pResVec[0].cls == it->cls)
            // {
                float iou_value = iou(p_pResVec[0], *it);
                if (iou_value > p_fNMSThresh)
                    it = p_pResVec.erase(it);
                else
                    it++;
            // }
            // else
            //     it++;
        }
        p_pResVec.erase(p_pResVec.begin());
    }
    return result;
}

void retina_get_output(std::vector<RetinaOutput>& retina_output, std::vector<Tensor*>& scores, std::vector<Tensor*>& bboxes, const std::vector<std::vector<CV::Rect> >& anchors, const std::vector<int>& strides, float thresh = 0.5, float nms_thresh = 0.4)
{
    retina_output.clear();
    int branch_num = scores.size();

    for (int i = 0; i < branch_num; i++)
    {
        int anchor_num = anchors[i].size();
        int h = scores[i]->shape()[2];
        int w = scores[i]->shape()[3];
        int step = h * w;
        float* score = scores[i]->host<float>();
        float* bbox = bboxes[i]->host<float>();
        int num_cls = scores[i]->shape()[1] / anchor_num;
        int stride = strides[i];

        for (int r = 0; r < h; r++)
        {
            for (int c = 0; c < w; c++, score++, bbox++)
            {
                for (int a = 0; a < anchor_num; a++)
                {
                    float max_prob = 0;
                    int cls = 0;
                    for (int k = 0; k < num_cls; k++)
                    {
                        float prob = score[(a * num_cls + k) * step];
                        if (prob > max_prob)
                        {
                            max_prob = prob;
                            cls = k;
                        }
                    }

                    if (max_prob > thresh)
                    {
                        // bbox decode
                        int bbox_start = 4 * a * step;
                        float bbox_x = bbox[bbox_start];
                        float bbox_y = bbox[bbox_start + step];
                        float bbox_w = bbox[bbox_start + 2 * step];
                        float bbox_h = bbox[bbox_start + 3 * step];

                        float anchor_w = anchors[i][a].width(); // * stride;
                        float anchor_h = anchors[i][a].height();// * stride;
                        float anchor_x = anchors[i][a].x() + c * stride;
                        float anchor_y = anchors[i][a].y() + r * stride;

                        bbox_x = bbox_x * anchor_w + anchor_x;
                        bbox_y = bbox_y * anchor_h + anchor_y;
                        bbox_w = expf(bbox_w) * anchor_w;
                        bbox_h = expf(bbox_h) * anchor_h;

                        if (bbox_w > 0 && bbox_h > 0)
                        {
                            RetinaOutput result;
                            result.cls = cls;
                            result.prob = max_prob;
                            result.bbox = CV::Rect::MakeXYWH(bbox_x - bbox_w / 2, bbox_y - bbox_h / 2, bbox_w, bbox_h);
                            retina_output.push_back(result);
                        }
                        
                    }
                }
            }
        }
    }

    // printf("nms %d\n", p_stResultVec.size());
    retina_output = retina_nms(retina_output, nms_thresh);
}

void run_net(std::shared_ptr<Interpreter>& net, MNN::Session* session, Matrix& trans, std::shared_ptr<ImageProcess>& pretreat, std::vector<std::vector<CV::Rect> >& anchors, std::vector<int>& strides, cv::Mat& img)
{
    int width = img.cols;
    int height = img.rows;
    int channel = img.channels();

    auto input = net->getSessionInput(session, NULL);
    for (int i = 0; i < input->dimensions(); ++i) {
        printf("%d, ", input->length(i));
    } printf("\n");

    int size_h       = input->height();
    int size_w       = input->width();

    float w_scale = width * 1.0f / size_w;
    float h_scale = height * 1.0f / size_h;

    MNN::Tensor* inputTensor = new Tensor(input, Tensor::CAFFE);

    trans.setScale((float)(width-1) / (size_w-1), (float)(height-1) / (size_h-1));
    pretreat->setMatrix(trans);
    pretreat->convert((uint8_t*)img.data, width, height, 0, inputTensor);

    input->copyFromHostTensor(inputTensor);
    net->runSession(session);
    // 将输出分成score 和 bbox
    auto outputs = net->getSessionOutputAll(session);
    std::vector<Tensor*> scores, bboxes;
    bool is_score = true;
    for (auto it = outputs.begin(); it != outputs.end(); it++)
    {
        Tensor* src_tensor = it->second;
        Tensor* dst_tensor = new Tensor(src_tensor, Tensor::CAFFE);
        for (int i = 0; i < src_tensor->dimensions(); ++i) {
            printf("%d, ", src_tensor->length(i));
        }
        printf("\n");
        src_tensor->copyToHostTensor(dst_tensor);

        if (is_score)
        {
            scores.push_back(dst_tensor);
            is_score = false;
        }
        else
        {
            bboxes.push_back(dst_tensor);
            is_score = true;
        }
    }

    sigmoid(scores);

    std::vector<RetinaOutput> retina_output;
    retina_get_output(retina_output, scores, bboxes, anchors, strides, 0.5, 0.1);

    printf("detected %d objs\n", retina_output.size());
    // cv::Mat new_img = img.clone();
    for (const auto& out : retina_output)
    {
        printf("cls: %d  prob: %f  bbox: %f %f %f %f\n", out.cls, out.prob, out.bbox.fLeft, out.bbox.fTop, out.bbox.fRight, out.bbox.fBottom);
        char text[64];
        sprintf(text, "%s: %.2f", VOC_NAMES[out.cls].c_str(), out.prob);
        cv::putText(img, text, cv::Point(out.bbox.x() * w_scale, out.bbox.y()* h_scale - 5), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0,0,255),1);
        cv::rectangle(img, cv::Rect(out.bbox.x() * w_scale, out.bbox.y() * h_scale, out.bbox.width() * w_scale, out.bbox.height() * h_scale), cv::Scalar(0,255,0), 2);
    }

    delete inputTensor;
    for (auto& tensor : scores)
        delete tensor;
    for (auto& tensor : bboxes)
        delete tensor;
}

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        MNN_PRINT("Usage: ./retinanet.out model.mnn input.jpg [word.txt]\n");
        return 0;
    }
    std::shared_ptr<Interpreter> net(Interpreter::createFromFile(argv[1]));
    ScheduleConfig config;
    // config.numThread = 2;
    config.type  = MNN_FORWARD_OPENCL; //MNN_FORWARD_OPENCL;
    printf("forward type: %d\n", config.type);
    // BackendConfig bnconfig;
    // bnconfig.memory = BackendConfig::Memory_High;
    // bnconfig.precision = BackendConfig::Precision_Low;
    // bnconfig.power = BackendConfig::Power_High;
    // config.backendConfig = &bnconfig;
    auto session = net->createSession(config);

    auto input = net->getSessionInput(session, NULL);
    auto shape = input->shape();
    shape[0]   = 1;
    // net->resizeTensor(input, shape);
    // net->resizeSession(session);
    auto outputs = net->getSessionOutputAll(session);

    // anchor
    std::vector<std::vector<CV::Rect> > anchors;
    int octave_base_scale = 4;
    int scales_per_octave = 3;
    std::vector<float> ratios{0.5, 1.0, 2.0};
    std::vector<int> strides{8, 16, 32, 64, 128};

    retina_config_anchor(anchors, octave_base_scale, scales_per_octave, ratios, strides);
    
    // for (auto it = outputs.begin(); it != outputs.end(); it++)
    // {
    //     Tensor* tensor = it->second;
    //     auto shape = tensor->shape();
    //     if (shape.size() == 4)
    //         printf("name: %s  shape: %d %d %d %d\n", it->first.c_str(), shape[0], shape[1], shape[2], shape[3]);
    //     if (shape.size() == 3)
    //         printf("name: %s  shape: %d %d %d\n", it->first.c_str(), shape[0], shape[1], shape[2]);
    // }
    std::vector<std::string> words;
    if (argc >= 4) {
        std::ifstream inputOs(argv[3]);
        std::string line;
        while (std::getline(inputOs, line)) {
            words.emplace_back(line);
        }
    }

    cv::Mat img;

    
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
    // int width, height, channel;
    
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
    float mean[3]     = {123.675, 116.28, 103.53};
    float normals[3] = {0.017f, 0.017f, 0.017f};
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

    
    /*
    std::ifstream ifs(argv[2]);
    std::string line;

    int num = 0;
    while (getline(ifs, line))
    {
        printf("%d/%d\n", num++, 4952);
        img = cv::imread(line);

        img = run_net(net, session, trans, pretreat, img);

        char save_path[64];
        sprintf(save_path, "voc_result/%05d.jpg", num);
        cv::imwrite(save_path, img);
    }
    */
    img = cv::imread(argv[2]);
    // run_net(net, session, trans, pretreat, anchors, strides, img);
    int width = img.cols;
    int height = img.rows;
    int channel = img.channels();

    input = net->getSessionInput(session, NULL);
    for (int i = 0; i < input->dimensions(); ++i) {
        printf("%d, ", input->length(i));
    } printf("\n");

    size_h       = input->height();
    size_w       = input->width();

    float w_scale = width * 1.0f / size_w;
    float h_scale = height * 1.0f / size_h;

    MNN::Tensor* inputTensor = new Tensor(input, Tensor::CAFFE);

    trans.setScale((float)(width-1) / (size_w-1), (float)(height-1) / (size_h-1));
    pretreat->setMatrix(trans);
    pretreat->convert((uint8_t*)img.data, width, height, 0, inputTensor);

    printf("input dimension: %d\n", input->getDimensionType());

    input->copyFromHostTensor(inputTensor);
    net->runSession(session);
    // 将输出分成score 和 bbox
    outputs = net->getSessionOutputAll(session);
    std::vector<Tensor*> scores, bboxes;
    bool is_score = true;
    for (auto it = outputs.begin(); it != outputs.end(); it++)
    {
        Tensor* src_tensor = it->second;
        Tensor* dst_tensor = new Tensor(src_tensor, Tensor::CAFFE);
        // for (int i = 0; i < src_tensor->dimensions(); ++i) {
        //     printf("%d, ", src_tensor->length(i));
        // }
        // printf("\n");
        printf("%s dimension: %d  type: %d\n", it->first.c_str(), src_tensor->getDimensionType(), src_tensor->getType().code);
        src_tensor->copyToHostTensor(dst_tensor);

        if (is_score)
        {
            scores.push_back(dst_tensor);
            is_score = false;
        }
        else
        {
            bboxes.push_back(dst_tensor);
            is_score = true;
        }
    }

    sigmoid(scores);

    std::vector<RetinaOutput> retina_output;
    retina_get_output(retina_output, scores, bboxes, anchors, strides, 0.5, 0.3);

    printf("detected %d objs\n", retina_output.size());
    // cv::Mat new_img = img.clone();
    for (const auto& out : retina_output)
    {
        printf("cls: %d  prob: %f  bbox: %f %f %f %f\n", out.cls, out.prob, out.bbox.fLeft, out.bbox.fTop, out.bbox.fRight, out.bbox.fBottom);
        char text[64];
        sprintf(text, "%s: %.2f", VOC_NAMES[out.cls].c_str(), out.prob);
        cv::putText(img, text, cv::Point(out.bbox.x() * w_scale, out.bbox.y()* h_scale - 5), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0,0,255),1);
        cv::rectangle(img, cv::Rect(out.bbox.x() * w_scale, out.bbox.y() * h_scale, out.bbox.width() * w_scale, out.bbox.height() * h_scale), cv::Scalar(0,255,0), 2);
    }

    delete inputTensor;
    for (auto& tensor : scores)
        delete tensor;
    for (auto& tensor : bboxes)
        delete tensor;
    cv::imwrite("retina_result.jpg", img);

    return 0;
}
