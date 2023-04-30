import torch
import torch.nn as nn
from mmseg.registry import MODELS
from mmseg.models.losses import CrossEntropyLoss


@MODELS.register_module()
class MultiLabelBCEWithLogitsLoss(CrossEntropyLoss):

    def __init__(self, use_sigmoid=False, reduction='mean', class_weight=None, loss_weight=1.0, ignore_index=None):
        super().__init__(use_sigmoid, reduction, class_weight, loss_weight, ignore_index)
        self.criterion = nn.BCEWithLogitsLoss(reduction=reduction)
    
    def forward(self, seg_logit, seg_label, ignore_index=None, **kwargs):
        # Change the shape of seg_logit and seg_label to (N, num_classes, H, W)
        seg_label = seg_label.permute(0, 2, 3, 1).contiguous().float()
        seg_logit = seg_logit.permute(0, 2, 3, 1).contiguous()
        num_classes = seg_logit.shape[-1]

        # Change the shape of seg_logit to (N * H * W, num_classes)
        seg_logit = seg_logit.view(-1, num_classes)

        # Change the shape of seg_label to (N * H * W, num_classes)
        seg_label = seg_label.view(-1, num_classes)

        #if self.class_weight is not None:
        #    weight = seg_label.new_tensor(self.class_weight)
        #    loss = self.criterion(seg_logit, seg_label, weight=weight)
        #else:
        loss = self.criterion(seg_logit, seg_label)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
