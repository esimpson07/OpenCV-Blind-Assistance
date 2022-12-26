'''
Code by Edward Simpson
12/23/2022
'''

from gtts import gTTS
import cv2 as cv
import numpy as np
import time
import math
import os

classes = ["background", "person", "bicycle", "car", "motorcycle",
  "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
  "unknown", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
  "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "unknown", "backpack",
  "umbrella", "unknown", "unknown", "handbag", "tie", "suitcase", "frisbee", "skis",
  "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
  "surfboard", "tennis racket", "bottle", "unknown", "wine glass", "cup", "fork", "knife",
  "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
  "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "unknown", "dining table",
  "unknown", "unknown", "toilet", "unknown", "tv", "laptop", "mouse", "remote", "keyboard",
  "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "unknown",
  "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" ]

colors = np.random.uniform(0, 255, size=(len(classes), 3))
cam = cv.VideoCapture(0)
pb  = 'frozen_inference_graph.pb'
pbt = 'ssd_inception_v2_coco_2017_11_17.pbtxt'
cvNet = cv.dnn.readNetFromTensorflow(pb,pbt)
currentTime = time.time()
pastTime = time.time()
delay = 2
height = 5.75
res = [480,640]
fov = [40,70]
tts = gTTS(text='Initializing...',lang="en")
fileName = "newAudioFile.mp3"
tts.save(fileName)

while True:
  currentTime = time.time()
  ret_val, img = cam.read()
  rows = img.shape[0]
  cols = img.shape[1]
  xlength = int(cols)
  cvNet.setInput(cv.dnn.blobFromImage(img, size=(200,200), swapRB=True, crop=False))
  cvOut = cvNet.forward()
  for detection in cvOut[0,0,:,:]:
    score = float(detection[2])
    if score > 0.5:
      idx = int(detection[1])
      if classes[idx] != ' ':
        tts = gTTS(text=classes[idx],lang="en")
        fileName = "newAudioFile.mp3"
        tts.save(fileName)
        left = detection[3] * cols
        top = detection[4] * rows
        right = detection[5] * cols
        bottom = detection[6] * rows
        pxfc = abs(((left + right) / 2) - res[0])
        distanceY = (0.5 * height) / math.tan(math.radians(0.5 * (bottom-top) / res[0]) * fov[0])
        print(distanceY)
        distanceX = (distanceY * math.atan(math.radians(fov[1] / 2)))
        print(distanceX)
        cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
        label = "{}: {:.2f}%".format(classes[idx],score * 100)
        y = top - 15 if top - 15 > 15 else top + 15
        cv.putText(img, label, (int(left), int(y)),cv.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)
        if(currentTime - pastTime >= delay):
          os.system(f"start {fileName}")
          pastTime = currentTime
  cv.imshow('OpenCV Detection', img)
  if cv.waitKey(1) == 27: 
    break
cam.release()
cv.destroyAllWindows()
