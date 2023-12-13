# [ICCV 2023] Small Object Detection via Coarse-to-fine Proposal Generation and Imitation Learning

## :loudspeaker: Introduction
This is the official implementation of our paper titled "Small Object Detection via Coarse-to-fine Proposal Generation and Imitation Learning", which has been accepted by [ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Yuan_Small_Object_Detection_via_Coarse-to-fine_Proposal_Generation_and_Imitation_Learning_ICCV_2023_paper.html).

## :ferris_wheel: Dependencies
 - CUDA 11.3
 - Python 3.8
 - PyTorch 1.10.0
 - TorchVision 0.11.0
 - mmcv-full 1.5.0
 - numpy 1.22.4

## :open_file_folder: Datasets
Our work is based on the large-scale small object detection benchmark **SODA**, which comprises two sub datasets **SODA-D** and **SODA-A**. Download the dataset(s) from corresponding links below.
 - SODA-D: [OneDrvie](https://nwpueducn-my.sharepoint.com/:f:/g/personal/gcheng_nwpu_edu_cn/EhXUvvPZLRRLnmo0QRmd4YUBvDLGMixS11_Sr6trwJtTrQ?e=PellK6); [BaiduNetDisk](https://pan.baidu.com/s/1aqmqkG_GzDKBTM_NK5ecqA?pwd=SODA)
 - SODA-A: [OneDrvie](https://nwpueducn-my.sharepoint.com/:f:/g/personal/gcheng_nwpu_edu_cn/EqJBjheHJXVOrMQWcr8dOt0BZJAfn1bkUSEQwIKHkVE0Vg?e=Hhcnoi); [BaiduNetDisk](https://pan.baidu.com/s/1G6x-hslv5C02WikZCzsNlA?pwd=SODA)

The data preparation for SODA differs slightly from that of conventional object detection datasets, as it requires the initial step of splitting the original images. 
Srcipts to obtain sub-images of SODA-D can be found at `tools/img_split`. For SODA-A, please refer to [SODA-mmrotate](https://github.com/shaunyuan22/SODA-mmrotate). More details about SODA please refer to the Dataset [Homepage](https://shaunyuan22.github.io/SODA/). 
<!-- and SODA-A can be found at [SODA-mmdetection](https://github.com/shaunyuan22/SODA-mmdetection) and -->



<!-- 
Moreover, this repository is build on MMDetection and MMrotate, please refer to [SODA-mmdetection](https://github.com/shaunyuan22/SODA-mmdetection) and [SODA-mmrotate](https://github.com/shaunyuan22/SODA-mmrotate) for the preparation of corresponding environment.
-->

## 🛠️ Install
This repository is build on **MMDetection 2.26.0**  which can be installed by running the following scripts. Please ensure that all dependencies have been satisfied before setting up the environment.
```
git clone https://github.com/shaunyuan22/CFINet
cd CFINet
pip install -v -e .
```
Moreover, please refer to [SODA-mmrotate](https://github.com/shaunyuan22/SODA-mmrotate) for MMRotate installation if you want to perform evaluation on the SODA-A dataset.

## 🚀 Training
 - Single GPU:
```
python ./tools/train.py ${CONFIG_FILE} 
```

 - Multiple GPUs:
```
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}
```

## 📈 Evaluation
 - Single GPU:
```
python ./tools/test.py ${CONFIG_FILE} ${WORK_DIR} --eval bbox
```

 - Multiple GPUs:
```
bash ./tools/dist_test.sh ${CONFIG_FILE} ${WORK_DIR} ${GPU_NUM} --eval bbox
```


## :trophy: Result
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

## 📚  Citation
Please cite our work if you find our work and codes helpful for your research.
```
@InProceedings{CFINet,
    author    = {Yuan, Xiang and Cheng, Gong and Yan, Kebing and Zeng, Qinghua and Han, Junwei},
    title     = {Small Object Detection via Coarse-to-fine Proposal Generation and Imitation Learning},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {6317-6327}
}

@ARTICLE{SODA,
  author={Cheng, Gong and Yuan, Xiang and Yao, Xiwen and Yan, Kebing and Zeng, Qinghua and Xie, Xingxing and Han, Junwei},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Towards Large-Scale Small Object Detection: Survey and Benchmarks}, 
  year={2023},
  volume={45},
  number={11},
  pages={13467-13488}
}
```

## :e-mail: Contact
If you have any problems about this repo or SODA benchmark, please be free to contact us at shaunyuan@mail.nwpu.edu.cn 😉

