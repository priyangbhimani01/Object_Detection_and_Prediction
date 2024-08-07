import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_hub as hub
import random
import math
from google.colab.patches import cv2_imshow

colorcodes = {}

model = hub.load("https://www.kaggle.com/models/google/faster-rcnn-inception-resnet-v2/TensorFlow1/faster-rcnn-openimages-v4-inception-resnet-v2/1").signatures['default']

def drawbox(image,ymin,xmin,ymax,xmax,namewithscore,color):
    im_height, im_width, _  = image.shape
    left,top,right,bottom = int(xmin*im_width), int(ymin*im_height), int(xmax*im_width),int(ymax*im_height)
    cv2.rectangle(image,(left,top),(right,bottom),color = color,thickness = 2)
    FONT_SCALE = 5e-3
    THICKNESS_SCALE = 4e-3
    width = right-left
    height = bottom-top
    TEXT_Y_OFFSET_SCALE = 1e-2
    cv2.rectangle(
        image,
        (left,top- int(height * 6e-2)),
        (right,top),
        color = color,
        thickness = -1

    )
    cv2.putText(
        image,
        namewithscore,
        (left,top-int(height * TEXT_Y_OFFSET_SCALE)),
        fontFace = cv2.FONT_HERSHEY_PLAIN,
        fontScale = min(width,height)* FONT_SCALE,
        thickness = math.ceil(min(width,height)* THICKNESS_SCALE),
        color = (255,255,255)
    )

def draw(image,boxes,classnames,scores):
    boxesidx = tf.image.non_max_suppression(boxes,scores,max_output_size = 10, iou_threshold = 0.5,score_threshold = 0.1)
    for i in boxesidx:
        ymin,xmin,ymax,xmax = tuple(boxes[i])
        classname = classnames[i].decode("ascii")
        if classname in colorcodes.keys():
            color = colorcodes[classname]
        else:
            c1 = random.randrange(0,255,30)
            c2 = random.randrange(0,255,25)
            c3 = random.randrange(0,255,50)
            colorcodes.update({classname: (c1,c2,c3)})
            color = colorcodes[classname]
        namewithscore = "{}:{}".format(classname,int(100*scores[i]))
        drawbox(image,ymin,xmin,ymax,xmax,namewithscore,color)

    return image

image = cv2.imread("image2.jpg")
image = cv2.resize(image,(800,600))
image2 = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
converted_img = tf.image.convert_image_dtype(image2,tf.float32)[tf.newaxis, ...]
detection = model(converted_img)
print(detection["detection_class_entities"])

# Convert detection results to numpy arrays
result = {key: value.numpy() for key, value in detection.items()}

# Draw bounding boxes on the image
imagewithboxes = draw(image, result['detection_boxes'], result['detection_class_entities'], result["detection_scores"])

# Display the image with bounding boxes using cv2_imshow
cv2_imshow(imagewithboxes)


