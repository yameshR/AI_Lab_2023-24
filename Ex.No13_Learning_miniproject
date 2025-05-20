# Ex.No: 13 Learning – Use Supervised Learning  
                                                                            
### REGISTER NUMBER : 212222060028

### AIM: 
To perform object detection on an image (e.g., cat.jpg) using OpenCV and a pre-trained MobileNet SSD model trained on the 
COCO dataset, and to display the detected object(s) with bounding boxes and labels.

### 🧠 **Algorithm:**

1. **Import Required Libraries**
   Import OpenCV and Colab-specific display tools.

2. **Read and Display the Image**
   Load an image using `cv2.imread()` and display using `cv2_imshow()`.

3. **Load Class Names**
   Load object class labels from the `coco.names` file into a list called `classNames`.

4. **Load Pre-trained Model**

   * Use `cv2.dnn_DetectionModel` to load the model configuration (`.pbtxt`) and weights (`.pb`).
   * Set input parameters like size, scale, mean, and color swapping.

5. **Detect Objects**

   * Use `net.detect()` to detect objects in the image with a confidence threshold of 0.7.
   * This returns class IDs, confidence scores, and bounding boxes.

6. **Draw Bounding Boxes and Labels**

   * For each detected object:

     * Draw a rectangle around it.
     * Display its class name using `cv2.putText()`.

7. **Show the Final Image**

   * Display the image with the detection results using `cv2_imshow()`.

### Program:
~~~
import cv2
from google.colab.patches import cv2_imshow
     

img = cv2.imread('cat.jpg')
     

cv2_imshow(img)

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
  classNames = f.read().rstrip('\n').split('\n') 

print(classNames)
     
['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
     

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
     
< cv2.dnn.Model 0x7940a4447950>

classIds, confs, bbox = net.detect(img, confThreshold=0.7)
     

print(classIds, bbox)
     
[17] [[101  67 239 322]]

for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x+w, y+h), color=(0,255,0), thickness=3)
    cv2.putText(img, classNames[classId-1], (x+10, y+30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
     

cv2_imshow(img)
~~~


### Output:
![CAT output](https://github.com/user-attachments/assets/dc974a7b-b57f-4fdd-8f54-9fd2aa549500)

### 📸 **Result:**

* The image `cat.jpg` is analyzed.
* The model detects a **cat** (class ID 17) with a high confidence.
* A **green rectangle** is drawn around the cat.
* The label **"cat"** is displayed near the detected region.
