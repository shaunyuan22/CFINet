dataset_type = 'SODADDataset'
data_root = '/data/SODA-D/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1200, 1200), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1200, 1200),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'divData/Annotations/train.json',
        img_prefix=data_root + 'divData/Images/',
        pipeline=train_pipeline,
        ori_ann_file=data_root + 'rawData/Annotations/train.json'
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'divData/Annotations/val.json',
        img_prefix=data_root + 'divData/Images/',
        pipeline=test_pipeline,
        ori_ann_file=data_root + 'rawData/Annotations/val_wo_ignore.json'
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'divData/Annotations/test.json',
        img_prefix=data_root + 'divData/Images/',
        pipeline=test_pipeline,
        ori_ann_file=data_root + 'rawData/Annotations/test_wo_ignore.json'
    ))
