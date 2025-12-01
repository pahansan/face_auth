import os

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

dataPath = 'dataSet'
faces, labels = [], []

for fname in os.listdir(dataPath):
    print(fname)
    if not fname.lower().endswith(('.jpg', '.png')):
        continue
    try:
        user_id = int(fname.split('.')[0].replace('face-', ''))
    except ValueError:
        continue

    img = Image.open(os.path.join(dataPath, fname)).convert('L')
    img_np = np.array(img, 'uint8')
    detections = cascade.detectMultiScale(img_np)
    for (x, y, w, h) in detections:
        faces.append(img_np[y:y + h, x:x + w])
        labels.append(user_id)

if not faces:
    raise RuntimeError('В dataSet не найдено ни одного лица')

X_tr, X_val, y_tr, y_val = train_test_split(
    faces, labels, test_size=0.2, random_state=42, stratify=labels)

recognizer.train(X_tr, np.array(y_tr))

y_pred = [recognizer.predict(f)[0] for f in X_val]
acc = accuracy_score(y_val, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(
    y_val, y_pred, average='weighted', zero_division=0)

print(f'Accuracy : {acc:.3f}')
print(f'Precision: {prec:.3f}')
print(f'Recall   : {rec:.3f}')
print(f'F1-score : {f1:.3f}')

MIN_ACC = 0.85
if acc >= MIN_ACC:
    os.makedirs('trainer', exist_ok=True)
    recognizer.write('trainer/trainer.yml')
    print('✅ Модель сохранена')
else:
    print('❌ Модель не сохранена (низкое качество)')
