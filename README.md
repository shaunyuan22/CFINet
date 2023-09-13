# [ICCV 2023] Small Object Detection via Coarse-to-fine Proposal Generation and Imitation Learning

## Introduction
This is the official implementation of our paper titled "Small Object Detection via Coarse-to-fine Proposal Generation and Imitation Learning", which has been accepted by ICCV 2023 and the preprint version has been submitted to [arXiv](https://arxiv.org/abs/2308.09534).

## Dependencies
 - CUDA 11.3
 - Python 3.8
 - PyTorch 1.10.0
 - TorchVision 0.11.0

## Datasets
Our work is based on the large-scale small object detection benchmark **SODA**, including two sub-dataset SODA-D and SODA-A. Please refer to the [Homepage](https://shaunyuan22.github.io/SODA/) of SODA for dataset downloading and performance evaluation. Moreover, this repository is build on MMDetection and MMrotate, please refer to [SODA-mmdetection](https://github.com/shaunyuan22/SODA-mmdetection) and [SODA-mmrotate](https://github.com/shaunyuan22/SODA-mmrotate) for the preparation of corresponding environment.

## Training

## Evaluation

## Result

    | **Method** | **Schedule** | **$AP$** | **$AP_{50}$** | **$AP_{75}$** | **$AP_T$** | **$AP_{eT}$** | **$AP_{rT}$** | **$AP_{gT}$** | **$AP_S$** |
    | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
    | Faster RCNN | $1 \times$ | 32.9 | 64.5 | 29.4 | 28.9 | 19.3 | 30.1 | 35.8 | 43.2 |
    | Cascade RCNN | $1 \times$ |35.7 | 64.6 | 33.8 | 31.2 | 20.4 | 32.5 | 39.0 | 46.9 |
    | RetinaNet | $1 \times$ | 29.2 | 58.0 | 25.3 | 25.0 | 15.7 | 26.3 | 31.8 | 39.6 |
    | FCOS | $1 \times$ | 28.7 | 55.1 | 26.0 | 23.9 | 11.9 | 25.6 | 32.8 | 40.9 |
    | RepPoints | $1 \times$ | 32.9 | 60.8 | 30.9 | 28.0 | 16.2 | 29.6 | 36.8 | 45.3 |
    | ATSS | $1 \times$ | 30.1 | 59.5 | 26.3 | 26.1 | 17.0 | 27.4 | 32.8 | 40.5 |
    | Deformable-DETR | $50e$ | 23.4 | 50.6 | 18.8 | 19.2 | 10.1 | 20.0 | 26.5 | 34.2 |
    | Sparse RCNN | $1 \times$ | 28.3 | 55.8 | 25.5 | 24.2 | 14.1 | 25.5 | 31.7 | 39.4 |

