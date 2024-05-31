import unittest
import sys
import cv2
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from VanillaOCR import OCR
from VanillaOCR import TextDetector, TextRecognizer, LabelDecoder
from VanillaOCR import drop_low_prob


class TestSimpleOCR_MainClass(unittest.TestCase):
    """
    测试 SimpleOCR 类的整体功能
    """
    def setUp(self):
        self.ocr = OCR()

    def test_png_1(self):
        """
        略有倾斜的表格。能跑通就行
        """
        image = cv2.imread("./test/test_images/test_1.png")
        texts, probs, boxes = self.ocr(image)
        self.assertTrue(
            s in texts[0] for s in ["创建", "修改", "表", "DROPTABLE", "视图"]
        )
        self.assertTrue(i > 0.9 for i in probs[0])


class TestSimpleOCR_OneByOne(unittest.TestCase):
    """
    测试 TextDetector、TextRecognizer、LabelDecoder 三个类协同
    """
    def setUp(self):
        self.text_detector = TextDetector()
        self.text_recognizer = TextRecognizer()
        char_list = self.text_recognizer.get_char_list()
        self.label_decoder = LabelDecoder(char_list, number=True, default_char_state=False)

    def test_png_2(self):
        """
        带有数字的表格。要求高精度识别纯数字片段
        """
        image = cv2.imread("./test/test_images/test_2.png")
        image = image[None, ...]
        boxes = self.text_detector(image)
        pred = self.text_recognizer(image, boxes)
        texts, probs = self.label_decoder(pred)

        texts, probs, boxes = drop_low_prob(texts, probs, boxes, 0.95)

        self.assertEqual(len(texts[0]), 15)
        self.assertTrue(
            s in texts[0] for s in ['5.8378', '14.7332', '11.4116', '8.4279', '12.2477']
        )


if __name__ == '__main__':
    unittest.main()
