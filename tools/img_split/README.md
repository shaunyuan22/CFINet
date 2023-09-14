# Preparing SODA-D Dataset

<!-- [DATASET] -->

```bibtex
@article{cheng2022towards,
  title={Towards large-scale small object detection: Survey and benchmarks},
  author={Cheng, Gong and Yuan, Xiang and Yao, Xiwen and Yan, Kebing and Zeng, Qinghua and Han, Junwei},
  journal={arXiv preprint arXiv:2207.14096},
  year={2022}
}
```


## download SODA-D dataset

The SODA-D dataset can be downloaded from [here](https://shaunyuan22.github.io/SODA/).

The data structure is as follows:

```none
mmdetection
├── mmdetection
├── tools
│   ├── img_split
│   │   ├── sodad_split.py
│   │   ├── split_configs
│   │   │   ├── split_train.json
│   │   │   ├── split_val.json
│   │   │   ├── split_test.json
│   │   ├── Images
│   │   ├── Annotations
│   │   │   ├── train.json
│   │   │   ├── val.json
│   │   │   ├── test.json
```

## split SODA-A dataset

The original images will be cropped to 800\*800 patches with the stride of 150.

```shell
python tools/img_split/sodad_split.py --cfgJson sodad_train.json 
```

If you want to visuzlize the annotations after split, please run the following script.

```shell
python tools/img_split/sodad_split.py --cfgJson sodad_train.json --isVis
```

## change configurations in split json files

Please change `oriImgDir`, `oriAnnDir` and `splDir` in json files before run the script.

