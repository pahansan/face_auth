#!/bin/bash

mkdir dataSet
mkdir trainer

wget https://github.com/spmallick/learnopencv/raw/refs/heads/master/AgeGender/opencv_face_detector_uint8.pb
wget wget https://github.com/spmallick/learnopencv/raw/refs/heads/master/AgeGender/opencv_face_detector.pbtxt
wget https://github.com/opencv/opencv/raw/refs/heads/4.x/data/haarcascades/haarcascade_frontalface_default.xml

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
