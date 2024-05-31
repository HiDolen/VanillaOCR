from VanillaOCR.onnx import ONNXModel
import cv2
import numpy as np
from einops import rearrange
from VanillaOCR.utils import get_configs


class TextDetector:
    def __init__(self):
        configs = get_configs()['TextDetector']

        self.model = ONNXModel(configs['model_path'], only_cpu=configs['only_cpu'])

        self.min_edge_after_resize = configs['min_edge_after_resize']
        self.length_base_number = configs['length_base_number']
        self.max_w_h_ratio = configs['max_w_h_ratio']
        self.mean = np.array(configs['mean'], dtype=np.float32)
        self.std = np.array(configs['std'], dtype=np.float32)

        self.binary_threshold = configs['binary_threshold']
        self.box_threshold = configs['box_threshold']
        self.unclip_ratio = configs['unclip_ratio']
        self.min_box_size = configs['min_box_size']
        self.dilation_kernel = np.array([[1, 1], [1, 1]])

    def pre_process(self, inputs):
        b, h, w, c = inputs.shape

        # resize
        scale = self.min_edge_after_resize / min(h, w)
        new_h = int(round(h * scale / self.length_base_number) * self.length_base_number)
        new_w = int(round(w * scale / self.length_base_number) * self.length_base_number)
        assert c * b < 512, 'CV_CN_MAX is 512. Batch size * channel should be less than 512.'
        inputs = rearrange(inputs, 'b h w c -> h w (c b)')
        inputs = cv2.resize(inputs, (new_w, new_h))
        inputs = rearrange(inputs, 'h w (c b) -> b h w c', b=b)

        # normalize
        inputs = inputs.astype(np.float32) / 255
        inputs = (inputs - self.mean) / self.std

        # transpose, [B, H, W, C] -> [B, C, H, W]
        inputs = inputs.transpose((0, 3, 1, 2))

        return inputs

    def post_process(self, outputs, original_height, original_width):
        def sort_box_points(box):
            axis_sums = np.sum(box, axis=1)
            axis_diffs = box[:, 1] - box[:, 0]

            top_left = box[np.argmin(axis_sums)]
            bottom_right = box[np.argmax(axis_sums)]

            top_right = box[np.argmin(axis_diffs)]
            bottom_left = box[np.argmax(axis_diffs)]

            return np.array([top_left, top_right, bottom_right, bottom_left])

        def get_box_score(box):
            """
            use binary mask to get box score
            """
            box = box.astype(np.int32)
            min_x = max(0, np.min(box[:, 0]))
            max_x = min(binary.shape[1], np.max(box[:, 0]))
            min_y = max(0, np.min(box[:, 1]))
            max_y = min(binary.shape[0], np.max(box[:, 1]))

            mask = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
            cv2.fillPoly(mask, [(box - np.array([min_x, min_y]))], 1)

            score = cv2.mean(outputs[0, min_y:max_y, min_x:max_x], mask)[0]

            return score

        def unclip(box):
            area = cv2.contourArea(box)
            perimeter = cv2.arcLength(box, True)
            distance = area * self.unclip_ratio / perimeter
            # signs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
            # expanded = box + distance * signs

            unit_vectors = []
            for i in range(4):
                vector = box[(i + 1) % 4] - box[i]
                unit_vector = vector / np.linalg.norm(vector)
                unit_vectors.append(unit_vector)
            new_box = np.zeros_like(box)
            for i in range(4):
                new_box[i] = box[i] + unit_vectors[i - 1] * distance
                new_box[i] = new_box[i] - unit_vectors[i] * distance

            expanded = new_box
            return expanded.astype(np.float32)

        outputs = outputs[:, 0, :, :]  # [B, 1, H, W] -> [B, H, W]
        b, h, w = outputs.shape

        # binary
        binary = outputs > self.binary_threshold

        # dilate
        binary = rearrange(binary, 'b h w -> h w b')
        dilate = cv2.dilate(binary.astype(np.uint8), self.dilation_kernel)
        if dilate.ndim == 2:
            dilate = dilate[:, :, None]
        dilate = rearrange(dilate, 'h w b -> b h w')

        # find contours
        contours_batch = []
        for image in dilate:
            contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours_batch.append(contours)

        # find boxes
        boxes_batch = []
        for contours in contours_batch:
            boxes = []
            for contour in contours:
                # get box
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                # drop box that are too small
                if min(rect[1]) < self.min_box_size:
                    continue
                box = np.array(box)
                # drop low score box
                if get_box_score(box) < self.box_threshold:
                    continue
                # sort box points. [top_left, top_right, bottom_right, bottom_left]
                box = sort_box_points(box)
                # unclip
                box = unclip(box)
                boxes.append(box)
            boxes = np.array(boxes)
            boxes_batch.append(boxes)

        # merge overlapping boxes
        # boxes_batch = [merge_boxes(boxes) for boxes in boxes_batch]

        # project boxes to original image
        boxes_batch = boxes_batch * np.array([original_width / w, original_height / h])

        return boxes_batch

    def __call__(self, inputs):
        """
        从大小一致的 np.ndarray 的一系列图片中检测文本框

        args:
            inputs: 输入图片，np.ndarray 格式，值范围 0~255，维度 [B, H, W, C]
        """
        b, h, w, c = inputs.shape
        w_h_ratio = w / h

        if w_h_ratio > self.max_w_h_ratio:
            padding = int(w / self.max_w_h_ratio - h)
            inputs = np.pad(inputs, ((0, 0), (0, padding), (0, 0), (0, 0)))
            h += padding

        inputs = self.pre_process(inputs)
        outputs = self.model(inputs)[0]
        boxes = self.post_process(outputs, h, w)

        return boxes

def merge_boxes(boxes):
    if len(boxes) == 0:
        return np.array([])

    # Perform non-maximum suppression
    keep = []
    while len(boxes) > 0:
        max_box = boxes[0]
        boxes = np.delete(boxes, 0, axis=0)

        overlap_indices = []
        merge = False
        for i, box in enumerate(boxes):
            if get_intersection_area(max_box, box) > 10:
                overlap_indices.append(i)
                merge = True

        if merge:
            merged_box = np.concatenate([[max_box], boxes[overlap_indices]], axis=0)
            merged_box = np.reshape(merged_box, (-1, 2))
            merged_box = cv2.minAreaRect(merged_box)
            merged_box = cv2.boxPoints(merged_box)
            merged_box = np.array(merged_box)
            keep.append(merged_box)
        else:
            keep.append(max_box)

        boxes = np.delete(boxes, overlap_indices, axis=0)

    return np.array(keep)


def get_intersection_area(box1, box2):
    box1 = box1.astype(np.int32)
    box2 = box2.astype(np.int32)

    min_x = max(np.min(box1[:, 0]), np.min(box2[:, 0]))
    max_x = min(np.max(box1[:, 0]), np.max(box2[:, 0]))
    min_y = max(np.min(box1[:, 1]), np.min(box2[:, 1]))
    max_y = min(np.max(box1[:, 1]), np.max(box2[:, 1]))

    if min_x >= max_x or min_y >= max_y:
        return 0

    mask_1 = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
    mask_2 = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
    cv2.fillPoly(mask_1, [box1 - np.array([min_x, min_y])], 1)
    cv2.fillPoly(mask_2, [box2 - np.array([min_x, min_y])], 1)
    mask = mask_1 + mask_2

    intersection = np.sum(mask == 2)
    if intersection != 0:
        a = 1

    return intersection
