from VanillaOCR import OCR
from VanillaOCR import TextDetector, TextRecognizer, LabelDecoder
from VanillaOCR.utils import extract_subImages, drop_low_prob
import cv2
import matplotlib.pyplot as plt

# ocr = SimpleOCR()
text_detector = TextDetector()
text_recognizer = TextRecognizer()
char_list = text_recognizer.get_char_list()
label_decoder = LabelDecoder(char_list)

image = cv2.imread(r"Z:\test.png")

# boxes = ocr.text_detector(image[None, ...])[0]
# pred = ocr.text_recognizer(image[None, ...], [boxes])
# result = ocr.label_decoder(pred)
boxes = text_detector(image[None, ...])[0]
pred = text_recognizer(image[None, ...], [boxes])
texts, probs = label_decoder(pred)

texts, probs, boxes = drop_low_prob(texts, probs, [boxes], 0.9)
boxes = boxes[0]


# 显示 sub_images 中的前 5 张图片
# sub_images = extract_subImages([image], [boxes])
# for i in range(5):
#     plt.imshow(sub_images[0][i][..., ::-1])
#     plt.show()

# 显示检测结果
for box in boxes:
    cv2.polylines(image, [box.astype(int)], True, (0, 255, 0), 1)

plt.imshow(image[..., ::-1])
plt.show()

# TODO 把可视化写到 utils 中
