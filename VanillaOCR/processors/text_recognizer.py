from VanillaOCR.onnx import ONNXModel
import cv2
import numpy as np
from einops import rearrange
from VanillaOCR.utils import get_configs
from itertools import islice
from VanillaOCR.utils import extract_subImages


class TextRecognizer:
    def __init__(self):
        configs = get_configs()['TextRecognizer']
        self.target_image_shape = configs['target_image_shape']
        self.batch_size = configs['batch_size']

        self.model = ONNXModel(configs['model_path'], only_cpu=configs['only_cpu'])

    def __call__(self, images_list, boxes_list):
        """
        args:
            images_list: List[np.ndarray]，list of original images
            boxes_list: List[np.ndarray]，list of detected boxes
        """
        # Extract sub-images
        subImages_batch = extract_subImages(images_list, boxes_list)
        # Flatten the batch
        count_each_batch = [len(subImages) for subImages in subImages_batch]
        subImages_batch = [image for batch in subImages_batch for image in batch]
        # Sort by width / height
        ratios = [image.shape[1] / image.shape[0] for image in subImages_batch]
        sorted_indices = np.argsort(ratios)
        # Inference
        outputs = []
        for i in range(0, len(subImages_batch), self.batch_size):
            batch_indices = sorted_indices[i: i + self.batch_size]
            batch_subImages = [subImages_batch[index] for index in batch_indices]
            batch_inputs = self._prepare_model_inputs(batch_subImages)
            batch_outputs = self.model(batch_inputs)[0]
            outputs.extend(batch_outputs)
        # Recover the batch order
        outputs = [outputs[i] for i in np.argsort(sorted_indices)]
        # Reshape the batch
        iterator = iter(outputs)
        outputs_batch = [list(islice(iterator, i)) for i in count_each_batch]

        return outputs_batch

    def _prepare_model_inputs(self, inputs):
        """
        args:
            inputs: List[np.ndarray], list of subimages
        """
        # resize
        target_h, target_w = self.target_image_shape
        heights = [image.shape[0] for image in inputs]
        widths = [image.shape[1] for image in inputs]
        new_widths = [int(target_h / h * w) for h, w in zip(heights, widths)]
        inputs = [cv2.resize(image, (w, target_h)) for image, w in zip(inputs, new_widths)]
        # normalize
        inputs = [image / 255 for image in inputs]
        inputs = [(image - 0.5) / 0.5 for image in inputs]
        # padding
        max_width = max(w for w in new_widths)
        inputs = [np.pad(image, ((0, 0), (0, max_width - image.shape[1]), (0, 0))) for image in inputs]
        # to numpy
        inputs = np.array(inputs).astype(np.float32)
        # transpose, [B, H, W, C] -> [B, C, H, w]
        inputs = inputs.transpose((0, 3, 1, 2))

        return inputs
    
    def get_char_list(self):
        meta_dict = self.model.session.get_modelmeta().custom_metadata_map
        return meta_dict["character"].splitlines()


