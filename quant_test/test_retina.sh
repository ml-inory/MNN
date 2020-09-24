#!/bin/bash
INPUT_FOLDER=VOCdevkit/VOC2007/JPEGImages

rm detection_output/unquant/retina/*
rm detection_output/quant/retina/*
./retinanet_quant.out models/unquant/retinanet_mbv2_voc.mnn $INPUT_FOLDER detection_output/unquant/retina
./retinanet_quant.out models/quant/retinanet_mbv2_voc.mnn $INPUT_FOLDER detection_output/quant/retina

echo "VOC Eval for unquantized Retina:  "
python3 voc_eval.py detection_output/unquant/retina
echo "VOC Eval for quantized Retina:  "
python3 voc_eval.py detection_output/quant/retina