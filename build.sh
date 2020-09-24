#!/bin/bash
mkdir -p build && cd build
cmake -DMNN_BUILD_DEMO=ON -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_QUANTOOLS=ON -DMNN_BUILD_QUANT_TEST=ON ..
make -j8
cp retinanet_quant.out ../quant_test/
cp ssd_quant.out ../quant_test/

