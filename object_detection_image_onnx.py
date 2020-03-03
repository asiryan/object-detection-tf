import onnxruntime as rt
import numpy as np
from PIL import Image, ImageDraw, ImageColor, ImageFont
import math
import matplotlib.pyplot as plt
import os
import sys

ROOT = os.getcwd()
WORK = os.path.join(ROOT, "ssd_mobilenet_v1_coco_2018_01_28")
IMAGES = os.path.join(ROOT, "images")
MODEL = "model"
os.makedirs(WORK, exist_ok=True)

# force tf2onnx to cpu
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
os.environ['MODEL'] = MODEL
os.environ['WORK'] = WORK

img = Image.open(os.path.join(IMAGES, "school.jpg"))
img_data = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
img_data = np.expand_dims(img_data.astype(np.uint8), axis=0)
print(img_data)
print(img_data.shape)
sess = rt.InferenceSession(os.path.join(WORK, MODEL + ".onnx"))

# we want the outputs in this order
outputs = ["num_detections:0", "detection_boxes:0", "detection_scores:0", "detection_classes:0"]
result = sess.run(outputs, {"image_tensor:0": img_data})
num_detections, detection_boxes, detection_scores, detection_classes = result
print(num_detections)
print(detection_classes)
print(detection_scores)

def draw_detection(draw, d, c):
    """Draw box and label for 1 detection."""
    width, height = draw.im.size
    print(width)
    print(height)
    # the box is relative to the image size so we multiply with height and width to get pixels.
    top = d[0] * height
    left = d[1] * width
    bottom = d[2] * height
    right = d[3] * width
    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(height, np.floor(bottom + 0.5).astype('int32'))
    right = min(width, np.floor(right + 0.5).astype('int32'))
    label = coco_classes[c]
    label_size = draw.textsize(label)
    text_origin = tuple(np.array([left + 1, top + 1]))
    color = ImageColor.getrgb("yellow")
    thickness = 0
    draw.rectangle([left + thickness, top + thickness, right - thickness, bottom - thickness], outline=color)
    draw.text(text_origin, label, fill=color, font=font)

coco_classes = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
}
font = ImageFont.truetype(
    "C:\\Windows\\Fonts\\Arial.ttf", 22)

batch_size = num_detections.shape[0]
draw = ImageDraw.Draw(img)
for batch in range(0, batch_size):
    for detection in range(0, int(num_detections[batch])):
        c = detection_classes[batch][detection]
        d = detection_boxes[batch][detection]
        draw_detection(draw, d, c)
        

plt.figure(figsize=(80, 40))
plt.axis('off')
plt.imshow(img)
plt.show()
