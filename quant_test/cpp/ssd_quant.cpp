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

#include "direct.h"
#include <vector>
#include <string>
#include <iostream>
#include "direct.h"
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/dir.h>

using namespace MNN;
using namespace MNN::CV;
using namespace std;

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
} SSDOutput;

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

void ssd_config_anchor(std::vector<std::vector<CV::Rect> >& anchors, float min_ratio, float max_ratio, const std::vector<int>& strides, const std::vector<std::vector<int> >& ratios, int input_size = 300, bool scale_major = false)
{
    anchors.clear();

    min_ratio *= 100;
    max_ratio *= 100;

    int num_levels = strides.size();
    int step = int(floor(max_ratio - min_ratio) / (num_levels - 2));
    std::vector<int> min_sizes, max_sizes;
    // VOC
    min_sizes.push_back(int(input_size * 10 / 100));
    max_sizes.push_back(int(input_size * 20 / 100));
    for (int r = int(min_ratio); r < int(max_ratio) + 1; r+=step)
    {
        min_sizes.push_back(int(input_size * r / 100));
        max_sizes.push_back(int(input_size * (r + step) / 100));
    }
    
    for (int k = 0; k < strides.size(); k++)
    {
        std::vector<CV::Rect> anchor;

        float offset = strides[k] * 0.5;
        float scales[2] = {1.f, sqrtf(max_sizes[k] * 1.0f / min_sizes[k])};
        int base_size = min_sizes[k];

        // printf("%d base_size: %d\n", k, base_size);
        std::vector<float> anchor_ratio;
        anchor_ratio.push_back(1.0f);
        for (const auto r : ratios[k])
        {
            anchor_ratio.push_back(1.0 / r);
            anchor_ratio.push_back(r);
        }

        CV::Rect tmp_anchor;
        if (scale_major)
        {
            for (int i = 0; i < anchor_ratio.size(); i++)
            {
                float h_ratio = sqrtf(anchor_ratio[i]);
                float w_ratio = 1.0f / h_ratio;
                for (int j = 0; j < 2; j++)
                {
                    anchor.push_back(CV::Rect::MakeXYWH(offset, offset, base_size * w_ratio * scales[j], base_size * h_ratio * scales[j]));
                }
                
                break;
            }
        }
        else
        {
            for (int i = 0; i < 2; i++)
            {
                float h_ratio, w_ratio;
                for (int j = 0; j < anchor_ratio.size(); j++)
                {
                    h_ratio = sqrtf(anchor_ratio[j]);
                    w_ratio = 1.0f / h_ratio;

                    anchor.push_back(CV::Rect::MakeXYWH(offset, offset, base_size * w_ratio * scales[i], base_size * h_ratio * scales[i]));

                    // printf("base_size: %d  w_ratio: %f  scale: %f\n", base_size, w_ratio, scales[i]);
                }
                h_ratio = sqrtf(anchor_ratio[0]);
                w_ratio = 1.0f / h_ratio;
                tmp_anchor = CV::Rect::MakeXYWH(offset, offset, base_size * w_ratio * scales[i+1], base_size * h_ratio * scales[i+1]);
                break;
            }
        }
        anchor.insert(anchor.begin() + 1, tmp_anchor);

        anchors.push_back(anchor);
    }
}

void softmax(std::vector<Tensor*>& scores, int num_cls = 21)
{
    for (auto& score : scores)
    {
        float* src = score->host<float>();
        int h, w, c;
        c = score->shape()[1];
        h = score->shape()[2];
        w = score->shape()[3];
        int step = h * w;
        int anchor_num = c / num_cls;

        // printf("anchor num: %d\n", anchor_num);
        for (int i = 0; i < h; i++)
        {
            for (int k = 0; k < w; k++)
            {
                for (int n = 0; n < c; n++)
                    src[n * step] = expf(src[n * step]);

                for (int a = 0; a < anchor_num; a++)
                {
                    float* sm = src + a * num_cls * step;
                    float exp_sum = 0.0f;
                    for (int n = 0; n < num_cls; n++)
                    {
                        exp_sum += sm[n * step];
                    }

                    for (int n = 0; n < num_cls; n++)
                    {
                        sm[n * step] /= exp_sum;
                    }
                }
                src++;
            }
        }
    }
}

float iou(const SSDOutput& box1, const SSDOutput& box2)
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

std::vector<SSDOutput> ssd_nms(std::vector<SSDOutput>& p_pResVec, float p_fNMSThresh = 0.3)
{
    std::vector<SSDOutput> result;

    std::sort(p_pResVec.begin(), p_pResVec.end(), [](SSDOutput lhs, SSDOutput rhs){ return lhs.prob > rhs.prob; });
    if (p_pResVec.size() > 1000)
    {
        p_pResVec.erase(p_pResVec.begin() + 1000, p_pResVec.end());
    }

    while (p_pResVec.size() > 0)
    {
        result.push_back(p_pResVec[0]);

        for(auto it = p_pResVec.begin() + 1; it != p_pResVec.end(); )
        {
            if (p_pResVec[0].cls == it->cls)
            {
                float iou_value = iou(p_pResVec[0], *it);
                if (iou_value > p_fNMSThresh)
                    it = p_pResVec.erase(it);
                else
                    it++;
            }
            else
                it++;
        }
        p_pResVec.erase(p_pResVec.begin());
    }
    return result;
}

void ssd_get_output(std::vector<SSDOutput>& ssd_output, std::vector<Tensor*>& scores, std::vector<Tensor*>& bboxes, const std::vector<std::vector<CV::Rect> >& anchors, const std::vector<int>& strides, float thresh = 0.5, float nms_thresh = 0.4)
{
    ssd_output.clear();
    int branch_num = scores.size();

    softmax(scores);

    for (int i = 0; i < branch_num; i++)
    {
        int anchor_num = anchors[i].size();
        
        int h = scores[i]->shape()[2];
        int w = scores[i]->shape()[3];
        int step = h * w;
        float* score = scores[i]->host<float>();
        float* bbox = bboxes[i]->host<float>();
        int num_cls = scores[i]->shape()[1] / anchor_num;
        // printf("num_cls: %d\n", num_cls);
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

                    if (cls < num_cls - 1 && max_prob > thresh)
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

                        // printf("prob: %f  anchor_w: %f  anchor_h: %f\n", max_prob, anchor_w, anchor_h);

                        bbox_x = 0.1 * bbox_x * anchor_w + anchor_x;
                        bbox_y = 0.1 * bbox_y * anchor_h + anchor_y;
                        bbox_w = expf(0.2 * bbox_w) * anchor_w;
                        bbox_h = expf(0.2 * bbox_h) * anchor_h;

                        if (bbox_w > 0 && bbox_h > 0)
                        {
                            SSDOutput result;
                            result.cls = cls;
                            result.prob = max_prob;
                            result.bbox = CV::Rect::MakeXYWH(bbox_x - bbox_w / 2, bbox_y - bbox_h / 2, bbox_w, bbox_h);
                            ssd_output.push_back(result);
                        }
                        
                    }
                }
            }
        }
    }

    // printf("nms %d\n", ssd_output.size());
    ssd_output = ssd_nms(ssd_output, nms_thresh);
}

int main(int argc, const char* argv[]) {
    if (argc < 4) {
        MNN_PRINT("Usage: ./ssd.out model.mnn input_folder output_folder\n");
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

    int branch_num = outputs.size() / 2;
    // anchor
    std::vector<std::vector<CV::Rect> > anchors;
    float min_ratio = 0.2f;
    float max_ratio = 0.95f;
    std::vector<int> strides{16, 30, 60, 100, 150, 300};
    std::vector<std::vector<int> > ratios{{2}, {2, 3}, {2, 3}, {2, 3}, {2}, {2}};

    ssd_config_anchor(anchors, min_ratio, max_ratio, strides, ratios);

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
    // auto inputImage = stbi_load(inputPatch, &width, &height, &channel, 4);
    // if (nullptr == inputImage) {
    //     MNN_ERROR("Can't open %s\n", inputPatch);
    //     return 0;
    // }
    // MNN_PRINT("origin size: %d, %d\n", width, height);
    Matrix trans;
    // Set transform, from dst scale to src, the ways below are both ok
    ImageProcess::Config pre_config;
    pre_config.filterType = BILINEAR;
    // mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
    float mean[3]     = {123.675, 116.28, 103.53};
    float normals[3] = {1.f, 1.f, 1.f};
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

        std::vector<SSDOutput> ssd_output;
        ssd_get_output(ssd_output, scores, bboxes, anchors, strides, 0.01, 0.3);
        string filename = filenames[i];
        filename = filename.substr(0, 6);
        for (auto& output : ssd_output)
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
