import torch
import logging
import numpy as np
import mmcv
import requests
import cv2

from ts.torch_handler.base_handler import BaseHandler
from sam.sam_inferencer import SamAutomaticMaskGenerator

logging.basicConfig(filename='/home/model-server/my_log.log', level=logging.INFO)


def is_box_close_to_border(tile_border, box, threshold=0):
    x, w, y, h = tile_border
    bx, by, bw, bh = box

    # distances between box sides and tile border
    left_dist = bx - x
    right_dist = x + w - (bx + bw)
    top_dist = by - y
    bottom_dist = y + h - (by + bh)

    # if is smaller than the threshold
    return any(d <= threshold for d in [left_dist, right_dist, top_dist, bottom_dist])


def aspect_ratio(box):
    w = box[2]
    h = box[3]
    if w > h:
        return w / h
    else:
        return h / w


def sliding_window_inference(img, mask_generator, step=256, window_size=512,
                             area_size_filter=[200, 3000], aspect=4):
    merged_boxes = []
    stability_score = []
    # Generate masks for each window
    for x in range(0, img.shape[1], step):
        for y in range(0, img.shape[0], step):
            logging.info('+1iter')
            crop = img[y:y + window_size, x:x + window_size]
            prompt_grid = adjust_grid(crop)

            if prompt_grid is None:
                continue

            logging.info('prompt_grid_sl_win: {}'.format(prompt_grid[0][0:5]))

            mask_generator.point_grids = prompt_grid

            masks = mask_generator.generate(crop)
            border = [y, y + window_size, x, x + window_size]
            for mask in masks:
                if area_size_filter[0] < mask['area'] < area_size_filter[1]:
                    # Shift the bbox to account for the crop
                    bbox = mask['bbox'] + np.array([x, y, 0, 0])
                    # if not is_box_close_to_border(border, bbox, threshold=1) and \
                    # aspect_ratio(bbox) < aspect:
                    if aspect_ratio(bbox) < aspect:
                        merged_boxes.extend([[int(i) for i in bbox.tolist()]])
                        # stability_score = mask['stability_score']

    return merged_boxes


# function to remove text from image
def remove_text(image, ocr_boxes):
    for box in ocr_boxes:
        image[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = 255
    return image


def skip_white_boxes(boxes, image, white_portion_th=0.95):
    """bbox in xywh format"""
    boxes_to_leave = []
    # convert image to gray
    image = mmcv.bgr2gray(image)
    for box in boxes:
        x, y, w, h = box
        box_image = image[int(y):int(y + h), int(x):int(x + w)]

        # get portion of white pixels
        white_portion = np.sum(box_image == 255) / (w * h)

        # if portion of white pixels is bigger then 95% return True
        if white_portion > white_portion_th:
            continue
        else:
            boxes_to_leave.append(box)

    return boxes_to_leave


def build_point_grid(n_per_side: int) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points


def adjust_grid(image, n_per_side=60):
    # Generate regular grid
    grid = build_point_grid(n_per_side)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    new_grid = []
    height, width = image.shape
    cell_width = width / n_per_side
    cell_height = height / n_per_side

    for i, point in enumerate(grid):
        patch = image[int(point[1] * height):int(point[1] * height + cell_height),
                int(point[0] * width):int(point[0] * width + cell_width)]

        # Find the black point inside the cropped patch using cv2.minMaxLoc()
        min_val, _, min_loc, _ = cv2.minMaxLoc(patch)
        black_point = (int(min_loc[0] + point[0] * width), int(min_loc[1] + point[1] * height))

        if min_val < 50:
            new_grid.append(black_point)

    # normalize grid
    new_grid = np.array(new_grid).astype('float64')

    if new_grid.size > 0:
        new_grid[:, 0] = new_grid[:, 0] / width
        new_grid[:, 1] = new_grid[:, 1] / height

        return [new_grid]
    else:
        return None


class MMsegHandler(BaseHandler):

    def initialize(self, context):
        logging.info('Start model init ...')
        properties = context.system_properties
        self.map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.map_location + ':' +
                                   str(properties.get('gpu_id')) if torch.cuda.
                                   is_available() else self.map_location)
        self.manifest = context.manifest
        self.model = SamAutomaticMaskGenerator(arch='custom')
        self.initialized = True

    def preprocess(self, data):
        images = []

        for row in data:
            request = row.get('data') or row.get('body')
            url = request['url']
            ocr_boxes = request['ocr_boxes']
            if not url.startswith("https://"):
                raise ValueError("Invalid S3 URL provided")

            response = requests.get(url)
            response.raise_for_status()

            image = mmcv.imfrombytes(response.content)

            if ocr_boxes:
                image = remove_text(image, ocr_boxes)
            # # crop bottom and right side of the image on 12.5%
            # image = image[:int(image.shape[0] * 0.875), :int(image.shape[1] * 0.875)]
            images.append(image)

        return images

    def inference(self, data, *args, **kwargs):
        step = 1024
        window_size = 1024
        # area_size_filter = [200, step*window_size/10]
        area_size_filter = [80, 100 * 50]  # limit the size to double doors size
        results = [sliding_window_inference(img, self.model, step=step, window_size=window_size,
                                            area_size_filter=area_size_filter) for img in data]

        # filter out white boxes
        results = [skip_white_boxes(boxes, image) for boxes, image in zip(results, data)]

        return results

    def postprocess(self, data):
        output = []

        for image_result in data:
            output.append({
                'bbox': image_result,
            })
        return output
