import torch
import torch.nn as nn
from mmcv.runner import force_fp32
from mmseg.core import add_prefix
from mmseg.registry import MODELS
from mmseg.models.losses import CrossEntropyLoss


@MODELS.register_module()
class MultiLabelBCEWithLogitsLoss(CrossEntropyLoss):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = nn.BCEWithLogitsLoss(*args, **kwargs)

    @force_fp32(apply_to=('seg_logit', 'seg_label'))
    def forward(self, seg_logit, seg_label, ignore_index=None):
        # Change the shape of seg_logit and seg_label to (N, num_classes, H, W)
        seg_logit = seg_logit.permute(0, 2, 3, 1).contiguous()
        seg_label = seg_label.permute(0, 2, 3, 1).contiguous()

        num_classes = seg_logit.shape[-1]

        # Change the shape of seg_logit to (N * H * W, num_classes)
        seg_logit = seg_logit.view(-1, num_classes)

        # Change the shape of seg_label to (N * H * W, num_classes)
        seg_label = seg_label.view(-1, num_classes)

        if self.class_weight is not None:
            weight = seg_label.new_tensor(self.class_weight)
            loss = self.criterion(seg_logit, seg_label, weight=weight)
        else:
            loss = self.criterion(seg_logit, seg_label)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss