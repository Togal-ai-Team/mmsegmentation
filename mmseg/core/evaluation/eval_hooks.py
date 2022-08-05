# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
import random

import torch.distributed as dist
from mmcv.runner import DistEvalHook as _DistEvalHook
from mmcv.runner import EvalHook as _EvalHook
from torch.nn.modules.batchnorm import _BatchNorm
import numpy as np
from torch import cat, unsqueeze, Tensor
from tensorboardX import SummaryWriter

class EvalHook(_EvalHook):
    """Single GPU EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
        pre_eval (bool): Whether to use progressive mode to evaluate model.
            Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self,
                 *args,
                 by_epoch=False,
                 efficient_test=False,
                 pre_eval=False,
                 log_tb=True,
                 log_wandb=True,
                 **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.pre_eval = pre_eval
        self.log_tb = log_tb
        self.log_wandb = log_wandb
        self.latest_results = None

        if efficient_test:
            warnings.warn(
                'DeprecationWarning: ``efficient_test`` for evaluation hook '
                'is deprecated, the evaluation hook is CPU memory friendly '
                'with ``pre_eval=True`` as argument for ``single_gpu_test()`` '
                'function')

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        from mmseg.apis import single_gpu_test
        results = single_gpu_test(
            runner.model, self.dataloader, show=False, pre_eval=self.pre_eval)
        if self.log_tb:
            import mmcv
            self.file_client = mmcv.FileClient(backend='disk')
            img_tensor_stack = None
            segmap_tensor_stack = None
            gt_tensor_stack = None

            self.tb_log_dir = osp.join(runner.work_dir, 'tf_logs')
            self.writer = SummaryWriter(self.tb_log_dir)
            for i, data in enumerate(self.dataloader):
                if random.random() < 0.95:
                    continue
                if img_tensor_stack is None:
                    img_tensor_stack = data['img'][0]
                    segmap_tensor_stack = unsqueeze(Tensor(results[i]), 0)
                    # annotations
                    ann_filename = data['img_metas'][0].data[0][0]['filename'].replace('images', 'annotations')
                    gt_bytes = self.file_client.get(ann_filename)
                    gt_tensor_stack = mmcv.imfrombytes(gt_bytes, flag='unchanged', backend='cv2').squeeze().astype(np.uint8)
                    gt_tensor_stack = unsqueeze(Tensor(gt_tensor_stack), 0)
                else:
                    # img
                    img_tensor_stack = cat((img_tensor_stack, data['img'][0]), 0)
                    # segmentation map
                    segmap_tensor_stack = cat((segmap_tensor_stack, unsqueeze(Tensor(results[i]), 0)), 0)
                    # annotation
                    ann_filename = data['img_metas'][0].data[0][0]['filename'].replace('images', 'annotations')
                    gt_bytes = self.file_client.get(ann_filename)
                    gt_tensor = mmcv.imfrombytes(gt_bytes, flag='unchanged', backend='cv2').squeeze().astype(np.uint8)
                    gt_tensor_stack = cat((gt_tensor_stack, unsqueeze(Tensor(gt_tensor), 0)), 0)
                if gt_tensor_stack.shape[0] > 8:
                    break

        segmap_tensor_stack = unsqueeze(segmap_tensor_stack, 1)
        gt_tensor_stack = np.expand_dims(gt_tensor_stack, 1)

        img_tensor_stack = np.asarray(img_tensor_stack*255, np.uint8)
        segmap_tensor_stack = np.asarray(segmap_tensor_stack*255, np.uint8)
        gt_tensor_stack = np.asarray(gt_tensor_stack, np.uint8)*255
        self.writer.add_images(
            'image/floorplan', img_tensor_stack, runner.iter
        )
        self.writer.add_images(
            'image/segmap', segmap_tensor_stack, runner.iter
        )
        self.writer.add_images(
            'image/annotation', gt_tensor_stack, runner.iter
        )
        if self.log_wandb:
            import wandb
            img_tensor_stack = np.moveaxis(img_tensor_stack, 1, 3)
            # concatenate floor plans, segmaps and ground truth
            for batch_idx in range(img_tensor_stack.shape[0]):
                floorplans_wandb = img_tensor_stack[batch_idx,...]
                segmap_wandb = np.repeat(np.expand_dims(segmap_tensor_stack[batch_idx,...], -1), 3, -1)
                gt_wandb = np.repeat(np.expand_dims(gt_tensor_stack[batch_idx, ...], -1), 3, -1)

                img_wandb = np.concatenate((floorplans_wandb, segmap_wandb[0,...], gt_wandb[0,...]), axis=1)
                floorplans_wandb = wandb.Image(img_wandb, caption="Floorplan")

                wandb.log({"floorplan": floorplans_wandb})

        self.latest_results = results
        runner.log_buffer.clear()
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)

        key_score = self.evaluate(runner, results)

        if self.save_best:
            self._save_ckpt(runner, key_score)


class DistEvalHook(_DistEvalHook):
    """Distributed EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
        pre_eval (bool): Whether to use progressive mode to evaluate model.
            Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self,
                 *args,
                 by_epoch=False,
                 efficient_test=False,
                 pre_eval=False,
                 **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.pre_eval = pre_eval
        self.latest_results = None
        if efficient_test:
            warnings.warn(
                'DeprecationWarning: ``efficient_test`` for evaluation hook '
                'is deprecated, the evaluation hook is CPU memory friendly '
                'with ``pre_eval=True`` as argument for ``multi_gpu_test()`` '
                'function')

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from mmseg.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect,
            pre_eval=self.pre_eval)
        self.latest_results = results
        runner.log_buffer.clear()

        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)

            if self.save_best:
                self._save_ckpt(runner, key_score)
