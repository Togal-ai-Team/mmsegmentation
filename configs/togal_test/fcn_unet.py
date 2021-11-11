_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/togal.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(test_cfg=dict(mode='whole'))
evaluation = dict(metric='mIoU')
