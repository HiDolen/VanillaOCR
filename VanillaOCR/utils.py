import yaml
import os
import numpy as np
import cv2

default_configs_path = os.path.join(os.path.dirname(__file__), "configs.yaml")

def get_configs(configs_path=default_configs_path):
    with open(configs_path, "r") as f:
        configs = yaml.safe_load(f)
    return configs


def extract_subImages(images_list, boxes_list):
    """
    Extract subimages from the original image according to the detected boxes.

    args:
        images_list: List[np.ndarray], list of original images
        boxes_list: List[np.ndarray], list of detected boxes
    """
    batch = []
    for image, boxes in zip(images_list, boxes_list):
        sub_images = []
        for box in boxes:
            box = box.astype(np.float32)
            width = max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3]))
            height = max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2]))
            dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(box, dst)
            sub_image = cv2.warpPerspective(
                image,
                M,
                (int(width), int(height)),
                borderMode=cv2.BORDER_REPLICATE,
                flags=cv2.INTER_CUBIC,
            )
            sub_images.append(sub_image)
        batch.append(sub_images)
    return batch

def get_char_type(char):
    """
    判断字符的类型。被 LabelDecoder 使用。

    args:
        char: str, 长度为 1 的字符串
    """
    if char == ' ' or char == 'blank':
        return 'reserved'

    if '\u0030' <= char <= '\u0039' or char == '.':
        return "number"  # 数字。包括小数点
    elif '\u4e00' <= char <= '\u9fff' or '\u3400' <= char <= '\u4dbf':
        return "zh"  # 中文
    elif '\u0041' <= char <= '\u005a':
        return "en_upper"  # 英文大写字母
    elif '\u0061' <= char <= '\u007a':
        return "en_lower"  # 英文小写字母
    elif (
        '\u3040' <= char <= '\u309f'
        or '\u30a0' <= char <= '\u30ff'
        or '\u31f0' <= char <= '\u31ff'
        or '\u4e00' <= char <= '\u9fff'
    ):
        return "ja"  # 日文
    elif '\uff00' <= char <= '\uffef':
        return "full_width"  # 全角字符
    elif '\u0020' <= char <= '\u007e':
        return "half_width"  # 半角字符
    else:
        return "others"

def drop_low_prob(texts, probs, boxes, drop_threshold):
    """
    丢弃概率过低的结果。用于 SimpleOCR 类的最终输出。

    args:
        texts: List[List[str]], 识别的文本。TextDecoder 的输出
        probs: List[List[float]], 识别的概率。TextDecoder 的输出
        boxes: List[np.ndarray], 文本框。TextDetector 的输出
        drop_threshold: float, 丢弃的概率阈值
    """
    new_texts, new_probs, new_boxes = [], [], []
    for text, prob, box in zip(texts, probs, boxes):
        new_text, new_prob, new_box = [], [], []
        for t, p, b in zip(text, prob, box):
            if p > drop_threshold:
                new_text.append(t)
                new_prob.append(p)
                new_box.append(b)
        new_texts.append(new_text)
        new_probs.append(new_prob)
        new_boxes.append(new_box)

    return new_texts, new_probs, new_boxes
