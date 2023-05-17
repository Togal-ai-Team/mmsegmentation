# Copyright (c) OpenMMLab. All rights reserved.
import base64
import os

import cv2
import mmcv
import torch
import numpy as np
from mmengine.model.utils import revert_sync_batchnorm
from ts.torch_handler.base_handler import BaseHandler

from mmseg.apis import inference_model, init_model


class MMsegHandler(BaseHandler):

    def initialize(self, context):
        properties = context.system_properties
        self.map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.map_location + ':' +
                                   str(properties.get('gpu_id')) if torch.cuda.
                                   is_available() else self.map_location)
        self.manifest = context.manifest

        model_dir = properties.get('model_dir')
        serialized_file = self.manifest['model']['serializedFile']
        checkpoint = os.path.join(model_dir, serialized_file)
        self.config_file = os.path.join(model_dir, 'config.py')

        self.model = init_model(self.config_file, checkpoint, self.device)
        self.model = revert_sync_batchnorm(self.model)
        self.initialized = True

    def preprocess(self, data):
        images = []

        for row in data:
            image = row.get('data') or row.get('body')
            if isinstance(image, str):
                image = base64.b64decode(image)
            image = mmcv.imfrombytes(image)
            images.append(image)

        return images

    def inference(self, data, *args, **kwargs):
        results = [inference_model(self.model, img) for img in data]
        return results

    def postprocess(self, data):
        output = []

        image_result = data[0]

        # assuming image_result.pred_sem_seg.data is a PyTorch tensor
        image_data = image_result.pred_sem_seg.data

        # if your data is in [0,1] range multiply by 255
        image_data = image_data * 255

        # move the channel dimension to the end and convert to uint8
        image_data = image_data.permute(1, 2, 0).byte()

        # transfer tensor to cpu and convert to numpy array for OpenCV
        image_data = image_data.cpu().numpy()

        _, buffer = cv2.imencode('.png', image_data)
        content = buffer.tobytes()
        output.append(content)

        return output
