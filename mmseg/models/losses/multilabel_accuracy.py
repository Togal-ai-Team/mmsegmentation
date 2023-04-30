# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn


def multilabel_accuracy(pred, target, thresh=0.5, ignore_index=None):
    """Calculate pixel-wise accuracy for multi-label segmentation according to the prediction and target.

    Args:
        pred (torch.Tensor): The model prediction, shape (N, C, W, H)
        target (torch.Tensor): The target of each prediction, shape (N, C, W, H)
        thresh (float, optional): Threshold to determine whether a class is present in a pixel or not.
            Default: 0.5.
        ignore_index (int | None): The label index to be ignored. Default: None

    Returns:
        float: The pixel-wise accuracy.
    """
    assert pred.ndim == target.ndim
    assert pred.size(0) == target.size(0)
    assert pred.size(1) == target.size(1)
    assert pred.size(2) == target.size(2)
    assert pred.size(3) == target.size(3)

    pred_labels = (pred > thresh).float()
    correct = (pred_labels == target).float()
    total_pixels = target.numel()
    total_correct = torch.sum(correct).float()
    eps = torch.finfo(torch.float32).eps
    accuracy = (total_correct + eps) / (total_pixels + eps)

    return accuracy
