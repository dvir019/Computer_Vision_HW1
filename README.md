# Surgical Tools Detection

## 1. Introduction
<p align="justify">
Object detection became very important in many fields, such as surgery, where most trainee surgeons learn by observing experts. Creating models which perform object detection on surgical videos can help surgeons learn more efficiently. In addition, accurate detection of surgical tools can help to ensure that procedures are performed safely and efficiently and can also help to prevent accidental misplacement of instruments. This project focuses in using YOLO-V5 model in order to perform tools detection.
</p>

## 2. Dataset
<p align="justify">
The dataset contains 1121 images of size 640×480. The images are taken from a video dataset that was created as part of another project. The original videos contain one person which performs a simulated procedure of placing three interrupted instrument-tied sutures on two opposing pieces of the material. Every procedure is performed using 3 tools: Scissors, Needle driver, and Forceps. Each of the tools can be used both in right and left hand, resulting in 6 possible combinations, and 2 additional for empty hands (no tool in use).
On top of the images, the dataset contains for each image 2 bounding boxes – one for the right hand and one for the left hand. The bounding boxes are given in a YOLO format. The dataset is split into 3 types of sets: Train, validation and test.

</p>

<p align="center" width="100%">
    <img width="80%" src="https://user-images.githubusercontent.com/30556126/228741730-8ee6c79f-4449-4ad8-85a1-088cf10ca657.png">
</p>

<p align="center" width="100%">
    <img width="50%" src="https://user-images.githubusercontent.com/30556126/228741970-5422ef83-39ae-4f7f-8ad7-377ff192d6f7.png">
    <img width="70%" src="https://user-images.githubusercontent.com/30556126/228741939-f0d8b031-2d0f-4200-ac2d-aaa05f82f8be.png">
</p>

## 3. Architecture
### 3.1. Grid Splitting
<p align="justify"> 
The YOLO model splits a given image into an S×S grid. If the center of an object falls into a grid cell, that grid cell is considered as “responsible” for detecting that object. Each grid cell predicts B bounding boxes and confidence scores for those boxes, which represent how much the model is confident that the box contains the object and how accurate it thinks the bounding box is.
Each bounding box consists of 5 predictions: x, y, w, h, and confidence.
</p>

<p align="center" width="100%">
    <img width="80%" src="https://user-images.githubusercontent.com/30556126/228742440-c9ed6f27-e4ae-4a0a-a9a0-242a3c488bbd.png">
</p>

## 3.2. Architecture Design
<p align="justify">
The YOLO model uses convolutional layers and fully connected layers: The convolutional layers are used to extract features from the image, and the fully connected layers predict the output probabilities and coordinates. It uses 1×1 reduction layers followed by 3×3 convolutional layers.
</p>

## 4. Expreiments
<p align="justify">
A several experminents was done. They different from one-another by the size of the model, the optimizer used for training and the the training data - some models where trained on an augmented data resulted by horizontally flipping the images.
</p>

<p align="center" width="100%">
    <img width="80%" src="https://user-images.githubusercontent.com/30556126/228743829-95bd2fdd-8218-48fb-9262-2464d30c3e4b.png">
</p>

## 5. Video Inference
<p align="justify">
I performed video inference using a model I trained. Given a video, I read it frame by frame, using cv2 module. Each frame is used as an input to the model, one frame at a time.
</p>

### 5.1. Missing Predictions
<p align="justify">
In some frames, the model didn’t include one of the hands in its prediction. In order to guess what the missing label is (denoted by -1), a window of a certain size was used, which include some of the previous predictions and some of the future ones. I calculated the most common label within the window, and assign the result as the missing label. I increased the size of the window and repeat the process incase that the majority label is -1.
In order to estimate the coordinates of the missing bounding box, I used simple kinematics: I calculated the velocity of the hand based on the last two frames and used it in order to estimate the location of the missing bounding box.
</p>

### 5.2. Multiple Predictions
<p align="justify">
In some frames the model predicted several different predictions for the same hand. In those cases, I chose the prediction with the higher confidence score.
</p>

### 5.3. Label Smoothing
<p align="justify">
In order to smooth the labels, I searched for labels which are different from all of the labels in the several past and future frames. I changed the predicted label of those lavels to the majority label withing the window.
</p>

### 5.4. Bounding Box Smoothing
<p align="justify">
I smoothed the bounding box coordinates by using a variant of exponential smoothing: For each bounding box, I calculated the weighted sum of the coordinates of the bounding boxes within a certain window, where the weight of each coordinates decreases exponentially with the further it is in time, except for the two neighbor frames, which have the same weight as the current frame’s coordinates.
</p>

<p align="center" width="100%">
    <img width="80%" src="https://user-images.githubusercontent.com/30556126/228746064-bb1bdb9a-c25d-4804-a4f6-95ecfcd923d5.png">
</p>
