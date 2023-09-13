# [ICCV 2023] Small Object Detection via Coarse-to-fine Proposal Generation and Imitation Learning

## Introduction
This is the official implementation of our paper titled "Small Object Detection via Coarse-to-fine Proposal Generation and Imitation Learning", which has been accepted by ICCV 2023 and the preprint version has been submitted to [arXiv](https://arxiv.org/abs/2308.09534).

## Dependencies
 - CUDA 11.3
 - Python 3.8
 - PyTorch 1.10.0
 - TorchVision 0.11.0
 - mmcv-full 1.5.0

## Datasets
Our work is based on the large-scale small object detection benchmark **SODA**, including two sub-dataset SODA-D and SODA-A. Please refer to the [Homepage](https://shaunyuan22.github.io/SODA/) of SODA for dataset downloading and performance evaluation. 
<!-- 
Moreover, this repository is build on MMDetection and MMrotate, please refer to [SODA-mmdetection](https://github.com/shaunyuan22/SODA-mmdetection) and [SODA-mmrotate](https://github.com/shaunyuan22/SODA-mmrotate) for the preparation of corresponding environment.
-->

## Install
This repository is build on MMDetection 2.23.0 which can be installed by running the following scripts. Before setting up the environment, please ensure that all dependencies have been satisfied.
```
git clone https://github.com/shaunyuan22/CFINet
cd CFINet
pip install -v -e .
```
Moreover, please refer to [SODA-mmrotate](https://github.com/shaunyuan22/SODA-mmrotate) if you want to perform evaluation on the SODA-A dataset.

## Training

## Evaluation

## Result
### Result on SODA-D
| **Method** | **Schedule** | **$AP$** | **$AP_{50}$** | **$AP_{75}$** | **$AP_{eS}$** | **$AP_{rS}$** | **$AP_{gS}$** | **$AP_N$** |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | 
| RetinaNet | $1 \times$ | 28.2 | 57.6 | 23.7 | 11.9 | 25.2 | 34.1 | 44.2 | 
| FCOS | $1 \times$ | 23.9 | 49.5 | 19.9 | 6.9 | 19.4 | 30.9 | 40.9 | 
| RepPoints | $1 \times$ | 28.0 | 55.6 | 24.7 | 10.1 | 23.8 | 35.1 | 45.3 | 
| ATSS | $1 \times$ | 26.8 | 55.6 | 22.1 | 11.7 | 23.9 | 32.2 | 41.3 | 
| YOLOX | $70e$ | 26.7 | 53.4 | 23.0 | 13.6 | 25.1 | 30.9 | 30.4 | 
| CornerNet | $2 \times$ | 24.6 | 49.5 | 21.7 | 6.5 | 20.5 | 32.2 | 43.8 | 
| CenterNet | $70e$ | 21.5 | 48.8 | 15.6 | 5.1 | 16.2 | 29.6 | 42.4 | 
| Deformable-DETR | $50e$ | 19.2 | 44.8 | 13.7 | 6.3 | 15.4 | 24.9 | 34.2 | 
| Sparse RCNN | $1 \times$ | 24.2 | 50.3 | 20.3 | 8.8 | 20.4 | 30.2 | 39.4 | 
| Faster RCNN | $1 \times$ | 28.9 | 59.4 | 24.1 | 13.8 | 25.7 | 34.5 | 43.0 | 
| Cascade RPN | $1 \times$ | 29.1 | 56.5 | 25.9 | 12.5 | 25.5 | 35.4 | 44.7 | 
| RFLA | $1 \times$ | 29.7 | 60.2 | 25.2 | 13.2 | 26.9 | 35.4 | 44.6 | 
| Ours | $1 \times$ | 30.7 | 60.8 | 26.7 | 14.7 | 27.8 | 36.4 | 44.6 | 

### Result on SODA-A
| **Method** | **Schedule** | **$AP$** | **$AP_{50}$** | **$AP_{75}$** | **$AP_{eS}$** | **$AP_{rS}$** | **$AP_{gS}$** | **$AP_N$** |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | 
| Rotated RetinaNet | $1 \times$ | 26.8 | 63.4 | 16.2 | 9.1 | 22.0 | 35.4 | 28.2 | 
| $S^2$ A-Net | $1 \times$ | 28.3 | 69.6 | 13.1 | 10.2 | 22.8 | 35.8 | 29.5 | 
| Oriented RepPoints | $1 \times$ | 26.3 | 58.8 | 19.0 | 9.4 | 22.6 | 32.4 | 28.5 |  
| DHRec | $1 \times$ | 30.1 | 68.8 | 19.8 | 10.6 | 24.6 | 40.3 | 34.6 | 
| Rotated Faster RCNN | $1 \times$ | 32.5 | 70.1 | 24.3 | 11.9 | 27.3 | 42.2 | 34.4 | 
| Gliding Vertex | $1 \times$ | 31.7 | 70.8 | 22.6 | 11.7 | 27.0 | 41.1 | 33.8 | 
| Oriented RCNN | $1 \times$ | 34.4 | 70.7 | 28.6 | 12.5 | 28.6 | 44.5 | 36.7 | 
| DODet | $1 \times$ | 31.6 | 68.1 | 23.4 | 11.3 | 26.3 | 41.0 | 33.5 | 
| Ours | $1 \times$ | 34.4 | 73.1 | 26.1 | 13.5 | 29.3 | 44.0 | 35.9 | 
