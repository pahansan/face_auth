#!/bin/bash

mkdir dataSet
mkdir trainer

wget --no-check-certificate "https://downloader.disk.yandex.ru/disk/6d115ccd5c5f0d8d512d7019a4e45873ad8153e95ffaba8f4024c0a8f0f429dc/692ca4fb/fKqInKw3d7bLFOeFnMGnhIIY8IUZ8FngwLK0ph4ubYlB0pksa6aec5gvHZH-BO9nn9tz3i23ahqT7b7FxBWxEwXdaTkkuWmMVpwP-ZZj-L6r8npumZHI4midPdWhecNq?uid=0&filename=opencv_face_detector_uint8.pb&disposition=attachment&hash=auN7LqnhR8OSn9Nw9S6yNzKe83tAqnEcbfURLkWvGXfKh2OeivptrN0sWMZYBV/Fq/J6bpmRyOJonT3VoXnDag%3D%3D&limit=0&content_type=application%2Foctet-stream&owner_uid=418113660&fsize=2727750&hid=bc3d227663bb9130f25ff141864c2b09&media_type=data&tknv=v3" -O opencv_face_detector_uint8.pb
wget --no-check-certificate "https://downloader.disk.yandex.ru/disk/7a71b6f02ec9a65fad61b69ebc600c38f4a3aeaaf8b22364628155a4641761a0/692ca4f1/hZKVK1sWsIlhLyxQK14sSElP4HADwdJkngQN8WcE5GmzOpiI9ZnB5Rq8lRs0ES3gctjmCepkz2_FuV6bE9FG8w%3D%3D?uid=0&filename=opencv_face_detector.pbtxt&disposition=attachment&hash=d5YzgeKgRAWDZnltDKOEDqlOrsnaC8Fosk9lWU5MfEnnRWm3yRrRvzL7a2JmHNISq/J6bpmRyOJonT3VoXnDag%3D%3D&limit=0&content_type=text%2Fplain&owner_uid=418113660&fsize=37272&hid=32b3ba6ea592150e0280ec4fd545c602&media_type=text&tknv=v3" -O opencv_face_detector.pbtxt
wget --no-check-certificate "https://downloader.disk.yandex.ru/disk/43a240adf75e02dff6e99af5e7b31870d26bd673dbf6c68caced2a7a0609360e/692cadfa/oK7-r_yEmv0csfUxCAJ-ZzVBZC0tkKAZzcWuA1TpZFAt_NEln2bdt8jhJQ-hQ84gQS5yskWhYz-bn-HBaW3E8g%3D%3D?uid=0&filename=haarcascade_frontalface_default.xml&disposition=attachment&hash=19lP%2BrW3N1Yhk5dtjr8rAk72g4EgC376GMKM/Oj2raOoPbqoprtwok7hMPTA%2BWSQq/J6bpmRyOJonT3VoXnDag%3D%3D&limit=0&content_type=text%2Fhtml&owner_uid=556733770&fsize=10217710&hid=49d40b17bab21f2d5421c0b737e91b2d&media_type=document&tknv=v3" -O haarcascade_frontalface_default.xml

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
