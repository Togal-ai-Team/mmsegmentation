import base64
import mmcv
import torch
import logging

from ts.torch_handler.base_handler import BaseHandler
from sam.sam_inferencer import SamAutomaticMaskGenerator

logging.basicConfig(filename='/home/model-server/my_log.log', level=logging.INFO)

class MMsegHandler(BaseHandler):

    def initialize(self, context):
        logging.info('Start model init ...')
        properties = context.system_properties
        self.map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.map_location + ':' +
                                   str(properties.get('gpu_id')) if torch.cuda.
                                   is_available() else self.map_location)
        self.manifest = context.manifest
        self.model = SamAutomaticMaskGenerator(arch='huge')
        self.initialized = True
        logging.info('Init complete ...')

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
        logging.info('Start inference')
        results = [self.model.generate(img) for img in data]
        logging.info('Prediction finished')
        logging.info(results)
        return results

    def postprocess(self, data):
        output = []

        for image_result in data:
            output.append({
                'bbox': [i['bbox'] for i in image_result],
                'area': [i['area'] for i in image_result],
                'stability_score': [i['stability_score'] for i in image_result],
            })
            logging.info(str(output))
        return output
