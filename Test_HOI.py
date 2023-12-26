from numpy.core.multiarray import result_type
import wget

# YOLOv3 weights
yolov3_weights_url = "https://github.com/patrick013/Object-Detection---Yolov3/blob/master/model/yolov3.weights?raw=true"
yolov3_weights_path = "yolov3.weights"
wget.download(yolov3_weights_url, yolov3_weights_path)

# YOLOv3 configuration file
yolov3_cfg_url = "https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true"
yolov3_cfg_path = "yolov3.cfg"
wget.download(yolov3_cfg_url, yolov3_cfg_path)

# COCO class names file
coco_names_url = "https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true"
coco_names_path = "coco.names"
wget.download(coco_names_url, coco_names_path)

print("Files downloaded successfully.")

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Status
Interaction=False

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load classes (coco.names contains the names of 80 classes)
with open("coco.names", "r") as f:
    classes = f.read().strip().split('\n')
# Reading Video
vidcap = cv2.VideoCapture("D:\Semester 7\ML\Project\VID_20231219_173544.mp4")
success,image = vidcap.read()
count=0
model=keras.models.load_model("D:\Semester 7\ML\Project\HOI_model_416.h5")
while success:
  clear_output()
  print(f"Interaction Detected at {round(count/30)} seconds") if Interaction else print(f"Interaction Not Detected at {round(count/30)} seconds")

  success,image = vidcap.read()
  count += 1
  if count%30==0:
     # Get image dimensions
    height, width = image.shape[:2]

    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    # Set input to the model
    net.setInput(blob)

    # Get output layer names
    output_layer_names = net.getUnconnectedOutLayersNames()

    # Run forward pass
    outs = net.forward(output_layer_names)

    # Initialize lists for detected objects' class IDs and confidence scores
    class_ids = []
    confidences = []
    # Post-processing: Get class IDs and confidence scores
    for out in outs:
      for detection in out:
          scores = detection[5:]
          class_id = np.argmax(scores)
          confidence = scores[class_id]

          if confidence > 0.7:  # Adjust this confidence threshold as needed
              if classes[class_id] == "person":
                  blob_reshaped=blob.reshape(3, 416, 416)
                  blob_transposed = np.transpose(blob_reshaped, (1, 2, 0))
                  final_image=blob_transposed.reshape(1, 416, 416, 3)
                  prob=model.predict(final_image).reshape(1)[0]
                  result = round(prob)
                  if result == 1:
                    Interaction=True
                    break
                  else :
                    Interaction=False
      if Interaction:
        break