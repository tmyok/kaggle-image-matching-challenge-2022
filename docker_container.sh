#!/bin/sh
# -*- coding: utf-8 -*-

docker run \
    -it \
    --gpus all \
    --name kaggle_IMC2022_$1 \
    --volume $(pwd)/:/home/work/ \
    --workdir /home/work/working/ \
    tmyok/pytorch:1.11.0-cuda11.6.2-cudnn8-opencv460-ubuntu20.04