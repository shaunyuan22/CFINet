# Preparing SODA-D Dataset

<!-- [DATASET] -->


## Download SODA-D dataset

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

## Split SODA-D dataset

The original images will be cropped to 800\*800 patches with the stride of 150.

```shell
python tools/img_split/sodad_split.py --cfgJson sodad_train.json 
```
Please change `oriImgDir`, `oriAnnDir` and `splDir` in json files before run the script. And if you want to visuzlize the annotations after split, please run the following script.

```shell
python tools/img_split/sodad_split.py --cfgJson sodad_train.json --isVis
```

## About evaluation

With regard to the evaluation, we'd like to bring two important points to your attention:
 - The evaluation is performed on the original images (**NOT ON** the splitted images).
 - The `ignore` regions will not be used in the evaluation phase.

Hence you need to filter `ignore` annotations in the `val.json` and `test.json` in the rawData directory to get `val_wo_ignore.json` and `test_wo_ignore.json` for final performance evaluation. Finally, you may have the following folder sturcture:

```none
SODA-D
├── rawData
│   ├── Images
│   ├── Annotations
│   │   ├── train.json
│   │   ├── train_wo_ignore.json
│   │   ├── val.json
│   │   ├── val_wo_ignore.json
│   │   ├── test.json
│   │   ├── test_wo_ignore.json
├── divData
│   ├── Images
│   │   ├── train
│   │   ├── val
│   │   ├── test
│   ├── Annotations
│   │   ├── train.json
│   │   ├── val.json
│   │   ├── test.json
```
