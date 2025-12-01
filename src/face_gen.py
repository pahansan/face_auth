import os
import time

import cv2

detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
offset   = 50
step     = 1
user_id  = input('Введите номер пользователя: ')

video    = cv2.VideoCapture(0)
if not video.isOpened():
    raise RuntimeError('Камера не открывается')

print('📹 Идёт запись… Нажмите Q для досрочного завершения')
buffer = []
start  = time.time()

while len(buffer) < 150:
    ok, frame = video.read()
    if not ok:
        break

    buffer.append(frame.copy())
    cv2.imshow('recording', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
video.release()
print(f'Записано {len(buffer)} кадров, обработка...')

saved = 0
h_max, w_max = buffer[0].shape[:2]

for idx in range(0, len(buffer), step):
    gray = cv2.cvtColor(buffer[idx], cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(100, 100))

    for (x, y, w, h) in faces:

        y1 = max(y - offset, 0)
        y2 = min(y + h + offset, h_max)
        x1 = max(x - offset, 0)
        x2 = min(x + w + offset, w_max)

        face_roi = gray[y1:y2, x1:x2]
        if face_roi.size == 0:
            continue

        cv2.imwrite(f'dataSet/face-{user_id}.{saved+1}.jpg', face_roi)
        saved += 1

print(f'✅ Сохранено {saved} лиц')
