_base_ = [
    '../_base_/datasets/sodad.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

dataset_type = 'SODADDataset'
data_root = '/data/SODA-D/'

model = dict(
    type='CenterNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        norm_eval=False,
        norm_cfg=dict(type='BN'),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='CTResNetNeck',
        in_channel=2048,
        num_deconv_filters=(1024, 512, 256),
        num_deconv_kernels=(4, 4, 4),
        use_dcn=False),
    bbox_head=dict(
        type='CenterNetHead',
        num_classes=9,
        in_channel=256,
        feat_channel=64,
        loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_wh=dict(type='L1Loss', loss_weight=0.1),
        loss_offset=dict(type='L1Loss', loss_weight=1.0)),
    train_cfg=None,
    test_cfg=dict(topk=100, local_maximum_kernel=3, max_per_img=100))

train_pipeline = [
    dict(
        type='RandomCenterCropPad',
        crop_size=(512, 512),
        ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True,
        test_pad_mode=None),
    dict(type='Resize', img_scale=(768, 768), keep_ratio=True),
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(
                type='RandomCenterCropPad',
                ratios=None,
                border=None,
                mean=[0, 0, 0],
                std=[1, 1, 1],
                to_rgb=True,
                test_mode=True,
                test_pad_mode=['logical_or', 31],
                test_pad_add_pix=1),
            dict(
                type='Collect',
                meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                           'scale_factor', 'flip', 'flip_direction',
                           'img_norm_cfg', 'border'),
                keys=['img'])
        ])
]

# Use RepeatDataset to speed up training
data = dict(
    train=dict(
        _delete_=True,
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'divData/Annotations/train.json',
            img_prefix=data_root + 'divData/Images/',
            pipeline=train_pipeline,
            ori_ann_file=data_root + 'rawData/Annotations/train.json')),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'divData/Annotations/val.json',
        img_prefix=data_root + 'divData/Images/',
        pipeline=test_pipeline,
        ori_ann_file=data_root + 'rawData/Annotations/val_wo_ignore.json'),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'divData/Annotations/test.json',
        img_prefix=data_root + 'divData/Images/',
        pipeline=test_pipeline,
        ori_ann_file=data_root + 'rawData/Annotations/test_wo_ignore.json'))

# optimizer
# Based on the default settings of modern detectors, the SGD effect is better
# than the Adam in the source code, so we use SGD default settings and
# if you use adam+lr5e-4, the map is 29.1.
optimizer = dict(type='SGD', lr=0.02 / 8, momentum=0.9, weight_decay=0.0001)   # bs = 2*6
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
# Based on the default settings of modern detectors, we added warmup settings.
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[9, 12])  # the real step is [9*5, 12*5]
runner = dict(max_epochs=14)  # the real epoch is14*5=70
evaluation = dict(interval=14, metric='bbox')
