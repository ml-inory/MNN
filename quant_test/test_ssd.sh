#!/bin/bash
INPUT_FOLDER=VOCdevkit/VOC2007/JPEGImages

rm detection_output/unquant/ssd/*
rm detection_output/quant/ssd/*
./ssd_quant.out models/unquant/ssd_mbv2_voc.mnn $INPUT_FOLDER detection_output/unquant/ssd
./ssd_quant.out models/quant/ssd_mbv2_voc.mnn $INPUT_FOLDER detection_output/quant/ssd

echo "VOC Eval for unquantized SSD:  "
python3 voc_eval.py detection_output/unquant/ssd
echo "VOC Eval for unquantized SSD:  "
python3 voc_eval.py detection_output/quant/ssd