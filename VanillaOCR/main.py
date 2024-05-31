from .processors.text_detector import TextDetector
from .processors.text_recognizer import TextRecognizer
from .processors.label_decoder import LabelDecoder
from .utils import get_configs, drop_low_prob
import numpy as np
import cv2
import os
import asyncio


class OCR:
    def __init__(self):
        configs = get_configs()["Overall"]
        self.drop_threshold_after_decode = configs["drop_threshold_after_decode"]

        self.text_detector = TextDetector()
        self.text_recognizer = TextRecognizer()
        char_list = self.text_recognizer.get_char_list()
        self.label_decoder = LabelDecoder(char_list)

    def __call__(self, images):
        """
        args:
            images: 支持 np.ndarray（维度 [H, W, C] 或 [B, H, W, C]）和 单个文件路径
        """
        images = self._load_images(images)
        boxes = self.text_detector(images)
        pred = self.text_recognizer(images, boxes)
        texts, probs = self.label_decoder(pred)

        texts, probs, boxes = drop_low_prob(texts, probs, boxes, self.drop_threshold_after_decode)

        return texts, probs, boxes

    def _load_images(self, images):
        """
        动态处理 np.ndarray、文件路径
        """
        if isinstance(images, np.ndarray):
            if images.ndim == 3:
                images = images[None, ...]
            return images
        elif isinstance(images, str):
            if os.path.isfile(images):
                return cv2.imread(images)[None, ...]
        else:
            raise ValueError('images 类型不支持')
