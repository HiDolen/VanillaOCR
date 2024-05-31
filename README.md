# VanillaOCR

简单，易拓展。

## 使用示例

直接使用 `VanillaOCR` 类：

```python
from VanillaOCR import OCR

ocr = OCR()
texts, probs, boxes = ocr("test.png")
print(texts[0])
```

分步识别：

```python
from VanillaOCR import TextDetector, TextRecognizer, LabelDecoder
from VanillaOCR import drop_low_prob

# 实例化 TextDetector TextRecognizer LabelDecoder
text_detector = TextDetector()
text_recognizer = TextRecognizer()
char_list = text_recognizer.get_char_list()
label_decoder = LabelDecoder(char_list)
# 读取与识别
image = cv2.imread("./test/test_images/test_2.png")
image = image[None, ...]
boxes = text_detector(image)
pred = text_recognizer(image, boxes)
texts, probs = label_decoder(pred)
# 丢弃 prob 低于 0.7 的结果
texts, probs, boxes = drop_low_prob(texts, probs, boxes, 0.7)

print(texts[0])
```