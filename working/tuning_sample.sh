#!/bin/sh

python3 exec_validation_data.py \
    --multirun \
    OUTPUT_RESULT=False \
    PRINT_INFO=False \
    'DKM.scale=choice(0.9, 1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5)' \
    'LoFTR.nms_ret_points=choice(500, 1000, 1500, 2000, 2500, 3000)' \
    'LoFTR.conf_min_thr=choice(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)' \
    'SE2LoFTR.nms_ret_points=choice(500, 1000, 1500, 2000, 2500, 3000)' \
    'SE2LoFTR.conf_min_thr=choice(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)' \
    'findFMat.input_num=choice(500, 1000, 1500, 2000, 2500, 3000)' \
    'findFMat.ransacReprojThreshold=choice(0.01, 0.05, 0.1, 0.15, 0.2, 0.25)' \
    'findFMat.confidence=choice(0.999, 0.9999, 0.99999, 0.999999)'