norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    bgr_to_rgb=True,
    pad_val=255,
    seg_pad_val=0,
    size=(448, 448))
model = dict(type='CascadeEncoderDecoder',
    num_stages=2,
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        bgr_to_rgb=True,
        pad_val=255,
        seg_pad_val=0,
        size=(448, 448)),
    pretrained=None,
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 4, 6, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    decode_head=[dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        out_channels=2,
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='MultiLabelBCEWithLogitsLoss')),
        dict(
            type='PointHead',
            in_channels=[64],
            in_index=[0],
            channels=256,
            num_fcs=3,
            coarse_pred_each_layer=True,
            dropout_ratio=-1,
            num_classes=2,
            align_corners=False,
            loss_decode=dict(
                type='MultiLabelBCEWithLogitsLoss'))
    ],

    train_cfg = dict(num_points=2048, oversample_ratio=3, importance_sample_ratio=0.75),
    test_cfg = dict(mode='whole'))
dataset_type = 'BaseSegDataset'
data_root = 'data'
crop_size = (720, 720)
train_pipeline = [
dict(type='LoadImageFromFile'),
dict(type='LoadMultiLabelAnnotations', reduce_zero_label=False),
dict(type='RandomFlip', prob=0.5),
dict(type='PhotoMetricDistortion'),
dict(type='RandomRotate', prob=0.7, angles=[90, 180, 270]),
dict(
    type='RandomCutOut',
    prob=0.2,
    n_holes=(3, 5),
    cutout_shape=(50, 50),
    fill_in=(255, 255, 255),
    seg_fill_in=0),
dict(type='PackSegInputs')
]
test_pipeline = [
dict(type='LoadImageFromFile'),
dict(type='LoadMultiLabelAnnotations', reduce_zero_label=False),
dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
dict(type='LoadImageFromFile', backend_args=None),
dict(
    type='TestTimeAug',
    transforms=[[{
        'type': 'Resize',
        'scale_factor': 0.5,
        'keep_ratio': True
    }, {
        'type': 'Resize',
        'scale_factor': 0.75,
        'keep_ratio': True
    }, {
        'type': 'Resize',
        'scale_factor': 1.0,
        'keep_ratio': True
    }, {
        'type': 'Resize',
        'scale_factor': 1.25,
        'keep_ratio': True
    }, {
        'type': 'Resize',
        'scale_factor': 1.5,
        'keep_ratio': True
    }, {
        'type': 'Resize',
        'scale_factor': 1.75,
        'keep_ratio': True
    }],
        [{
            'type': 'RandomFlip',
            'prob': 0.0,
            'direction': 'horizontal'
        }, {
            'type': 'RandomFlip',
            'prob': 1.0,
            'direction': 'horizontal'
        }], [{
            'type': 'LoadAnnotations'
        }], [{
            'type': 'PackSegInputs'
        }]])
]
train_dataloader = dict(
batch_size = 32,
num_workers = 4,
persistent_workers = True,
sampler = dict(type='InfiniteSampler', shuffle=True),
dataset = dict(
    type='BaseSegDataset',
    data_root='data',
    data_prefix=dict(
        img_path='images/train', seg_map_path='annotations/train'),
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadMultiLabelAnnotations', reduce_zero_label=False),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(type='RandomRotate', prob=0.7, angles=[90, 180, 270]),
        dict(
            type='RandomCutOut',
            prob=0.2,
            n_holes=(3, 5),
            cutout_shape=(50, 50),
            fill_in=(255, 255, 255),
            seg_fill_in=0),
        dict(type='PackSegInputs')
    ],
    img_suffix='.png'))
val_dataloader = dict(
batch_size = 1,
num_workers = 4,
persistent_workers = True,
sampler = dict(type='DefaultSampler', shuffle=False),
dataset = dict(
    type='BaseSegDataset',
    data_root='data',
    data_prefix=dict(
        img_path='images/test', seg_map_path='annotations/test'),
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadMultiLabelAnnotations', reduce_zero_label=False),
        dict(type='PackSegInputs')
    ],
    img_suffix='.png'))
test_dataloader = dict(
batch_size = 1,
num_workers = 4,
persistent_workers = True,
sampler = dict(type='DefaultSampler', shuffle=False),
dataset = dict(
    type='BaseSegDataset',
    data_root='data',
    data_prefix=dict(
        img_path='images/test', seg_map_path='annotations/test'),
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadMultiLabelAnnotations', reduce_zero_label=False),
        dict(type='PackSegInputs')
    ],
    img_suffix='.png'))
val_evaluator = dict(type='MultiIoUMetric', iou_metrics=['mIoU'])
test_evaluator = dict(type='MultiIoUMetric', iou_metrics=['mIoU'])
default_scope = 'mmseg'
env_cfg = dict(
cudnn_benchmark = True,
mp_cfg = dict(mp_start_method='fork', opencv_num_threads=0),
dist_cfg = dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
type = 'SegLocalVisualizer',
vis_backends = [dict(type='LocalVisBackend')],
name = 'visualizer',
save_dir = 'output')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = 'work_dirs/segformer_mit-b2_8xb2-160k_ade20k-512x512/iter_48000.pth'
resume = True
tta_model = dict(type='SegTTAModel')
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(
type = 'OptimWrapper',
optimizer = dict(
    type='AdamW', lr=6e-05, betas=(0.9, 0.999), weight_decay=0.01),
paramwise_cfg = dict(
    custom_keys=dict(
        pos_block=dict(decay_mult=0.0),
        norm=dict(decay_mult=0.0),
        head=dict(lr_mult=10.0))))
param_scheduler = [
dict(
    type='LinearLR', start_factor=1e-06, by_epoch=False, begin=0,
    end=1500),
dict(
    type='PolyLR',
    eta_min=0.0,
    power=1.0,
    begin=1500,
    end=160000,
    by_epoch=False)
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=50000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
timer = dict(type='IterTimerHook'),
logger = dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
param_scheduler = dict(type='ParamSchedulerHook'),
checkpoint = dict(type='CheckpointHook', by_epoch=False, interval=2000),
sampler_seed = dict(type='DistSamplerSeedHook'),
visualization = dict(type='SegVisualizationHook', draw=True))
launcher = 'none'
work_dir = './work_dirs/segformer_mit-b2_8xb2-160k_ade20k-512x512_pointrend'

