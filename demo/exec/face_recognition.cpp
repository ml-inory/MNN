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

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

using namespace MNN;
using namespace MNN::CV;

void get_feature(Tensor* feat_tensor, std::vector<float>& feature)
{
    float* src = feat_tensor->host<float>();
    int feat_size = feat_tensor->shape()[1];
    feature.resize(feat_size);
    
    float norm = 0;
    for (int i = 0; i < feat_size; i++)
    {
        norm += src[i] * src[i];
    }
    norm = sqrtf(norm);

    for (int i = 0; i < feat_size; i++)
    {
        feature[i] = src[i] / norm;
    }
}


int main(int argc, const char* argv[]) {
    if (argc < 3) {
        MNN_PRINT("Usage: ./retinanet.out model.mnn input.jpg [word.txt]\n");
        return 0;
    }
    std::shared_ptr<Interpreter> net(Interpreter::createFromFile(argv[1]));
    ScheduleConfig config;
    config.type  = MNN_FORWARD_AUTO;
    // BackendConfig bnconfig;
    // bnconfig.precision = BackendConfig::Precision_Low;
    // config.backendConfig = &bnconfig;
    auto session = net->createSession(config);

    auto input = net->getSessionInput(session, NULL);
    auto shape = input->shape();
    shape[0]   = 1;
    net->resizeTensor(input, shape);
    net->resizeSession(session);

    cv::Mat img = cv::imread(argv[2]);

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
    width = img.cols;
    height = img.rows;
    channel = img.channels();
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
    trans.setScale((float)(width-1) / (size_w-1), (float)(height-1) / (size_h-1));
#endif
    ImageProcess::Config pre_config;
    pre_config.filterType = BILINEAR;
    // mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
    float mean[3]     = {127.5, 127.5, 127.5};
    float normals[3] = {0.0078125, 0.0078125, 0.0078125};
    // float mean[3]     = {127.5f, 127.5f, 127.5f};
    // float normals[3] = {0.00785f, 0.00785f, 0.00785f};
    ::memcpy(pre_config.mean, mean, sizeof(mean));
    ::memcpy(pre_config.normal, normals, sizeof(normals));
    pre_config.sourceFormat = BGR;
    pre_config.destFormat   = RGB;

    std::shared_ptr<ImageProcess> pretreat(ImageProcess::create(pre_config));
    pretreat->setMatrix(trans);
    // pretreat->convert((uint8_t*)inputImage, width, height, 0, input);
    // stbi_image_free(inputImage);

    float w_scale = width * 1.0f / size_w;
    float h_scale = height * 1.0f / size_h;

    pretreat->convert((uint8_t*)img.data, width, height, 0, input);

    net->runSession(session);
    // 将输出分成score 和 bbox
    auto output = net->getSessionOutput(session, NULL);
    auto outputTensor = new Tensor(output, Tensor::CAFFE);
    output->copyToHostTensor(outputTensor);

    std::vector<float> feature;
    get_feature(outputTensor, feature);

    delete outputTensor;
    return 0;
}
