import os
import time

import cv2
import numpy as np
from imutils import rotate_bound

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
faceNet = cv2.dnn.readNet(faceModel, faceProto)

offset = 30
nFrames = 100
step = 5
user_id = input('Введите номер пользователя: ')
name = input('Введите имя пользователя: ')

os.makedirs('dataSet', exist_ok=True)
os.makedirs('names', exist_ok=True)

with open(f'names/{user_id}.txt', 'w', encoding='utf-8') as f:
    f.write(name)

def detect_faces_dnn(frame, conf_threshold=0.7):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 [104, 117, 123], True, False)
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            faces.append((x1, y1, x2 - x1, y2 - y1))
    return faces

def preprocess_face(face_img, target_size=(200, 200)):
    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img

    gray = cv2.equalizeHist(gray)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    resized = cv2.resize(gray, target_size)

    return resized

def augment_face(face_img):
    augmented = []

    augmented.append(face_img)

    for alpha in [0.7, 0.9, 1.1, 1.3]:
        bright = cv2.convertScaleAbs(face_img, alpha=alpha, beta=0)
        augmented.append(bright)

    for angle in [-15, -10, -5, 5, 10, 15]:
        rotated = rotate_bound(face_img, angle)
        rotated = cv2.resize(rotated, (200, 200))
        augmented.append(rotated)

    flipped = cv2.flip(face_img, 1)
    augmented.append(flipped)

    for ksize in [(3, 3), (5, 5)]:
        blurred = cv2.GaussianBlur(face_img, ksize, 0)
        augmented.append(blurred)

    return augmented

video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Ошибка: Не удалось открыть камеру")
    exit()

print(f'Сбор данных для пользователя {name} (ID: {user_id})')
print('Двигайте головой: влево/вправо, вверх/вниз')
print('Меняйте выражение лица: улыбка, нейтральное, серьезное')
print('Нажмите Q для досрочного завершения')
print(f'Сбор данных автоматически завершится после {nFrames} кадров')

buffer = []
saved_count = 0
frame_count = 0

while len(buffer) < nFrames:
    ret, frame = video.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        print("Ошибка чтения кадра")
        break

    frame_count += 1

    faces = detect_faces_dnn(frame, conf_threshold=0.7)

    display_frame = frame.copy()
    if faces:
        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(display_frame, f"Face detected", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if frame_count % 3 == 0:
            buffer.append(frame.copy())
    else:
        cv2.putText(display_frame, "No face detected", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(display_frame, f"Frames: {len(buffer)}/{nFrames}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow('Data Collection - Face Detection', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
video.release()

print(f'\nСобрано {len(buffer)} кадров. Обработка...')

saved_faces = 0

for idx, frame in enumerate(buffer[::step]):
    faces = detect_faces_dnn(frame, conf_threshold=0.7)

    for (x, y, w, h) in faces:
        y1 = max(y - offset, 0)
        y2 = min(y + h + offset, frame.shape[0])
        x1 = max(x - offset, 0)
        x2 = min(x + w + offset, frame.shape[1])

        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            continue

        processed_face = preprocess_face(face_roi)
        augmented_faces = augment_face(processed_face)

        for aug_face in augmented_faces:
            filename = f'dataSet/face-{user_id}.{saved_faces+1:04d}.jpg'
            cv2.imwrite(filename, aug_face)
            saved_faces += 1

    if (idx + 1) % 5 == 0 and idx != 0:
        print(f'  Обработано кадров: {idx + 1}/~{len(buffer)//step}...')

print(f'\nСохранено {saved_faces} изображений лица')
print(f'Данные сохранены в папке dataSet/')
print(f'Имя пользователя сохранено в names/{user_id}.txt')
