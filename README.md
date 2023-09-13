# [ICCV 2023] Small Object Detection via Coarse-to-fine Proposal Generation and Imitation Learning

## Introduction
This is the official implementation of our paper titled "Small Object Detection via Coarse-to-fine Proposal Generation and Imitation Learning", which has been accepted by ICCV 2023 and the preprint version has been submitted to [arXiv](https://arxiv.org/abs/2308.09534).

## Dependencies
 - CUDA 11.3
 - Python 3.8
 - PyTorch 1.10.0
 - TorchVision 0.11.0

## Datasets
Our work is evaluated on the large-scale benchmark for small object detection: SODA, including two sub-dataset SODA-D and SODA-A. Please refer to the [Homepage](https://shaunyuan22.github.io/SODA/) of SODA for dataset downloading and performance evaluation. Moreover, this repository is build on MMDetection and MMrotate, please refer to [SODA-mmdetection](https://github.com/shaunyuan22/SODA-mmdetection) and [SODA-mmrotate](https://github.com/shaunyuan22/SODA-mmrotate) for the preparation of corresponding environment.

