import torch
import logging
import numpy as np
import mmcv
import requests
import cv2
import ast

from ts.torch_handler.base_handler import BaseHandler
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

logging.basicConfig(filename='/home/model-server/my_log.log', level=logging.INFO)


def filter_boxes_containing_another_boxes(boxes):
    """
        Experimental!
        Filters out boxes that contain other boxes inside. If box is pretty big it could contain
        smaller parts inside.

        Args:
            boxes (list): List of bounding boxes represented as [x, y, width, height].

        Returns:
            list: Filtered list of bounding boxes.

    """
    filtered_boxes = []
    for i, box in enumerate(boxes):
        x1, y1, w1, h1 = box
        is_inside = False
        for j, box2 in enumerate(boxes):
            if i != j:
                x2, y2, w2, h2 = box2
                if x1 <= x2 and y1 <= y2 and x1 + w1 >= x2 + w2 and y1 + h1 >= y2 + h2:
                    is_inside = True
                    break
        if not is_inside:
            filtered_boxes.append(box)
    return filtered_boxes


def filter_box_outliers(bounding_boxes, median_threshold=10):
    """
        Filters out outliers from a list of bounding boxes based on their width and height.

        Args:
            bounding_boxes (list): List of bounding boxes represented as [x, y, width, height].
            median_threshold (int, optional): Threshold multiplier for filtering outliers.

        Returns:
            list: Filtered list of bounding boxes.

    """
    # Compute the median width and height of the bounding boxes
    widths = np.array([box[2] for box in bounding_boxes])
    heights = np.array([box[3] for box in bounding_boxes])
    median_width = np.median(widths)
    median_height = np.median(heights)

    # Filter out the bounding boxes that are significantly larger than the median size
    filtered_boxes = [box for box in bounding_boxes
                      if (box[2] <= median_threshold * median_width) and
                      (box[3] <= median_threshold * median_height)]

    return filtered_boxes


def sliding_window_inference(img, mask_generator, step=256, window_size=512,
                             area_size_filter=[200, 3000]):
    try:
        total_patches = -(-img.shape[1] // step) * -(-img.shape[0] // step)
        patch_count = 0

        merged_boxes = []
        # Generate masks for each window
        for x in range(0, img.shape[1], step):
            for y in range(0, img.shape[0], step):
                patch_count += 1
                logging.info('Processing patches: {}/{}'.format(patch_count, total_patches))
                crop = img[y:y + window_size, x:x + window_size]
                prompt_grid = adjust_grid(crop)

                if prompt_grid is None:
                    continue

                mask_generator.point_grids = prompt_grid

                masks = mask_generator.generate(crop)
                for mask in masks:
                    if area_size_filter[0] < mask['area'] < area_size_filter[1]:
                        # Shift the bbox to account for the crop
                        bbox = mask['bbox'] + np.array([x, y, 0, 0])
                        # if aspect_ratio(bbox) < aspect:
                        merged_boxes.extend([[int(i) for i in bbox.tolist()]])
        logging.info('Inference is done')

        return merged_boxes

    except Exception as e:
        logging.error(f"Error during sliding window inference: {str(e)}")


# function to remove text from image
def remove_text(image, ocr_boxes):
    try:
        ocr_boxes = None if ocr_boxes == ("" or "None") else ast.literal_eval(ocr_boxes)

        if ocr_boxes:
            ocr_boxes = ocr_boxes['features']
            height, width, _ = image.shape

            # get boxes from geojson in absolute coordinates and plot them
            text_boxes = []
            for i in range(len(ocr_boxes)):
                vertices = ocr_boxes[i]['geometry']['coordinates'][0]
                if ocr_boxes[i]['geometry']['type'] == 'Polygon':
                    min_x = min(vertex[0] for vertex in vertices)
                    min_y = min(vertex[1] for vertex in vertices)
                    max_x = max(vertex[0] for vertex in vertices)
                    max_y = max(vertex[1] for vertex in vertices)
                    text_boxes.append([min_x, min_y, max_x, max_y])

                elif ocr_boxes[i]['geometry']['type'] == 'MultiPolygon':
                    for polygon in vertices:
                        min_x = min(vertex[0] for vertex in polygon)
                        min_y = min(vertex[1] for vertex in polygon)
                        max_x = max(vertex[0] for vertex in polygon)
                        max_y = max(vertex[1] for vertex in polygon)
                        text_boxes.append([min_x, min_y, max_x, max_y])

            # conert boxes to absolute coordinates
            text_boxes = np.array(text_boxes)
            text_boxes[:, 0] *= width
            text_boxes[:, 1] *= height
            text_boxes[:, 2] *= width
            text_boxes[:, 3] *= height

            for box in text_boxes:
                image[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = 255

        return image

    except Exception as e:
        logging.error(f"Error of OCR processing: {str(e)}")


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


def adjust_grid(image, n_per_side=60, brightness_threshold=50):
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

        if min_val < brightness_threshold:
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
        try:
            logging.info('Start model init ...')
            properties = context.system_properties
            self.map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.device = torch.device(self.map_location + ':' +
                                       str(properties.get('gpu_id')) if torch.cuda.
                                       is_available() else self.map_location)
            self.manifest = context.manifest
            sam = sam_model_registry['vit_b'](
                checkpoint='/home/model-server/model-store/sam_finetuned_synth.pth')
            sam.to(device='cuda')
            self.model = SamAutomaticMaskGenerator(sam,
                                                   pred_iou_thresh=0.7,
                                                   stability_score_thresh=0.85
                                                   )
            self.initialized = True
        except Exception as e:
            logging.error(f"Error during model init: {str(e)}")

    def preprocess(self, data):
        try:
            images = []

            for row in data:
                request = row.get('data') or row.get('body')
                url = request['url']
                logging.info(f"Processing image: {url}")

                ocr_boxes = request['ocr_boxes']
                if not url.startswith("https://"):
                    raise ValueError("Invalid S3 URL provided")

                response = requests.get(url)
                response.raise_for_status()

                image = mmcv.imfrombytes(response.content)

                if ocr_boxes:
                    image = remove_text(image, ocr_boxes)
                images.append(image)

                return images
        except Exception as e:
            logging.error(f"Error during preprocessing: {str(e)}")

    def inference(self, data, *args, **kwargs):
        try:
            step = 720
            window_size = 720
            # area_size_filter = [200, step*window_size/10]
            area_size_filter = [20, 100 * 50]  # limit the size to double doors size
            results = [sliding_window_inference(img, self.model, step=step, window_size=window_size,
                                                area_size_filter=area_size_filter) for img in data]
            # filter out white boxes
            results = [skip_white_boxes(boxes, image) for boxes, image in zip(results, data)]
            # filter out boxes that are too big
            results = [filter_box_outliers(boxes) for boxes in results]
            # filter out boxes containing other boxes
            results = [filter_boxes_containing_another_boxes(boxes) for boxes in results]

            return results

        except Exception as e:
            logging.error(f"Error during inference: {str(e)}")
            return []

    def postprocess(self, data):
        output = []

        for image_result in data:
            output.append({
                'bbox': image_result,
            })
        return output
