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
#include <string>
#include <iostream>
#include "direct.h"
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/dir.h>

// #include <MNN/AutoTime.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

using namespace MNN;
using namespace MNN::CV;
using namespace std;

//所有目录
void getCurrentDir(string path, vector<string>& full_files, vector<string>& filenames) {
    struct dirent *entry;
    DIR *dir = opendir(path.c_str());

    if (dir == NULL) {
        return;
    }
    while ((entry = readdir(dir)) != NULL) {
        string filename(entry->d_name);
        if (filename.find(".jpg") != filename.npos)
        {
            full_files.push_back(path + "/" + filename);
            filenames.push_back(filename);
        }
    }
    closedir(dir);
}

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
        }
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


int main(int argc, const char* argv[]) {
    if (argc < 4) {
        MNN_PRINT("Usage: ./retinanet.out model.mnn input_folder output_folder\n");
        return 0;
    }
    std::shared_ptr<Interpreter> net(Interpreter::createFromFile(argv[1]));
    ScheduleConfig config;
    config.numThread = 4;
    config.type  = MNN_FORWARD_AUTO; //MNN_FORWARD_OPENCL;
    // BackendConfig bnconfig;
    // bnconfig.memory = BackendConfig::Memory_High;
    // bnconfig.precision = BackendConfig::Precision_Low;
    // bnconfig.power = BackendConfig::Power_High;
    // config.backendConfig = &bnconfig;
    auto session = net->createSession(config);

    auto input = net->getSessionInput(session, NULL);
    auto shape = input->shape();
    shape[0]   = 1;
    net->resizeTensor(input, shape);
    net->resizeSession(session);
    auto outputs = net->getSessionOutputAll(session);

    // anchor
    std::vector<std::vector<CV::Rect> > anchors;
    int octave_base_scale = 4;
    int scales_per_octave = 3;
    std::vector<float> ratios{0.5, 1.0, 2.0};
    std::vector<int> strides{8, 16, 32, 64, 128};

    retina_config_anchor(anchors, octave_base_scale, scales_per_octave, ratios, strides);

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

    // 读取输入文件夹
    std::string input_folder(argv[2]);
    std::string output_folder(argv[3]);
    vector<string> input_files, filenames;
    getCurrentDir(input_folder, input_files, filenames);

    for (int i = 0; i < input_files.size(); i++)
    {
        if (i % 100 == 0)
            printf("Process %d/%d\n", i, input_files.size());
        img = cv::imread(input_files[i]);
        if (img.empty())
        {
            printf("Cannot read %s\n", input_files[i].c_str());
            break;
        }
        int width = img.cols;
        int height = img.rows;
        int channel = img.channels();

        float w_scale = width * 1.0f / size_w;
        float h_scale = height * 1.0f / size_h;

        MNN::Tensor* inputTensor = new Tensor(input, Tensor::CAFFE);

        trans.setScale((float)(width-1) / (size_w-1), (float)(height-1) / (size_h-1));
        pretreat->setMatrix(trans);
        pretreat->convert((uint8_t*)img.data, width, height, 0, inputTensor);

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
        retina_get_output(retina_output, scores, bboxes, anchors, strides, 0.01, 0.3);
        string filename = filenames[i];
        filename = filename.substr(0, 6);
        for (auto& output : retina_output)
        {
            output.bbox.fLeft *= w_scale;
            output.bbox.fTop *= h_scale;
            output.bbox.fRight *= w_scale;
            output.bbox.fBottom *= h_scale;

            string cls = VOC_NAMES[output.cls];
            ofstream ofs(output_folder + "/" + cls + ".txt", ios::app);
            ofs << filename << " " << output.prob << " " << output.bbox.fLeft << " " << output.bbox.fTop << " " << output.bbox.fRight << " " << output.bbox.fBottom << endl;
        }

        delete inputTensor;
        for (auto& tensor : scores)
            delete tensor;
        for (auto& tensor : bboxes)
            delete tensor;
    }

    return 0;
}
