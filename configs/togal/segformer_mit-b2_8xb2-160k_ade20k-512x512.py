_base_ = ['../segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py']
# dataset settings
dataset_type = 'BaseSegDataset'
data_root = 'data'
crop_size = (720, 720)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadMultiLabelAnnotations', reduce_zero_label=False),
    dict(type='RandomMultiMaskFlip', prob=0.6, direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='RandomCutOut', prob=0.2, n_holes=(3, 5), cutout_shape=(40, 40), fill_in=(255, 255, 255), seg_fill_in=0),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomRotate', prob=0.5, angles=[90, 180, 270]),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadMultiLabelAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix='.png',
        data_prefix=dict(
            img_path='images/train', seg_map_path='annotations/train'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix='.png',
        data_prefix=dict(
            img_path='images/test',
            seg_map_path='annotations/test'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='MultiIoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

resume = True
load_from = 'work_dirs/segformer_mit-b2_8xb2-160k_ade20k-512x512/iter_48000.pth'
# model settings
model = dict(
    pretrained=None,
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 4, 6, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))
