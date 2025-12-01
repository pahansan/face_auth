import os
import time
from collections import Counter

import cv2
import imutils
import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

os.makedirs('trainer', exist_ok=True)
os.makedirs('reports', exist_ok=True)

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
faceNet = cv2.dnn.readNet(faceModel, faceProto)

def load_names():
    names = {}
    names_dir = 'names'
    if os.path.exists(names_dir):
        for fname in os.listdir(names_dir):
            if fname.endswith('.txt'):
                try:
                    user_id = int(fname.split('.')[0])
                    with open(os.path.join(names_dir, fname), 'r', encoding='utf-8') as f:
                        name = f.read().strip()
                    names[user_id] = name
                except:
                    continue
    return names

def preprocess_face(image, target_size=(200, 200)):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    gray = cv2.equalizeHist(gray)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    resized = cv2.resize(gray, target_size)

    return resized

print("Загрузка данных...")
dataPath = 'dataSet'
faces = []
labels = []
image_files = []

for fname in os.listdir(dataPath):
    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_files.append(fname)

print(f"Найдено {len(image_files)} изображений")

for fname in image_files:
    try:
        parts = fname.split('.')
        user_id = int(parts[0].replace('face-', ''))

        img_path = os.path.join(dataPath, fname)
        img = Image.open(img_path).convert('L')
        img_np = np.array(img, 'uint8')

        processed = preprocess_face(img_np)

        faces.append(processed)
        labels.append(user_id)

    except Exception as e:
        print(f"Ошибка обработки {fname}: {e}")
        continue

print(f"Загружено {len(faces)} лиц, {len(set(labels))} пользователей")

label_counter = Counter(labels)
print("\nРаспределение данных по классам:")
for label_id, count in label_counter.items():
    print(f"  ID {label_id}: {count} изображений")

if len(faces) == 0:
    raise RuntimeError('❌ В dataSet не найдено ни одного лица')

print("\nРазделение данных...")
X = np.array(faces)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Обучающая выборка: {len(X_train)} изображений")
print(f"  Тестовая выборка: {len(X_test)} изображений")

print("\nОбучение моделей...")

param_sets = [
    {'radius': 1, 'neighbors': 8, 'grid_x': 16, 'grid_y': 16},
    {'radius': 1, 'neighbors': 8, 'grid_x': 10, 'grid_y': 10},
    {'radius': 1, 'neighbors': 8, 'grid_x': 8, 'grid_y': 8},
    {'radius': 1, 'neighbors': 4, 'grid_x': 8, 'grid_y': 8},
]

best_model = None
best_accuracy = 0
best_params = None
results = []

for i, params in enumerate(param_sets):
    print(f"\n  Модель {i+1}/{len(param_sets)}:")
    print(f"    Параметры: {params}")

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=params['radius'],
        neighbors=params['neighbors'],
        grid_x=params['grid_x'],
        grid_y=params['grid_y'],
        threshold=70
    )

    recognizer.train(X_train, y_train)

    y_pred = []
    confidences = []

    for face in X_test:
        label, confidence = recognizer.predict(face)
        y_pred.append(label)
        confidences.append(confidence)

    accuracy = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )

    results.append({
        'params': params,
        'accuracy': accuracy,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'model': recognizer
    })

    print(f"    Accuracy:  {accuracy:.3f}")
    print(f"    Precision: {prec:.3f}")
    print(f"    Recall:    {rec:.3f}")
    print(f"    F1-score:  {f1:.3f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = recognizer
        best_params = params

print("\n" + "="*50)
print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ")
print("="*50)

print(f"\nЛучшая модель:")
print(f"  Параметры: {best_params}")
print(f"  Accuracy:  {best_accuracy:.3f}")

MIN_ACCURACY = 0.80

if best_accuracy >= MIN_ACCURACY:
    model_path = 'trainer/trainer.yml'
    best_model.save(model_path)

    params_path = 'trainer/model_params.npy'
    np.save(params_path, best_params)

    metrics = {
        'accuracy': best_accuracy,
        'n_classes': len(set(labels)),
        'n_samples': len(faces),
        'train_size': len(X_train),
        'test_size': len(X_test),
    }
    metrics_path = 'trainer/metrics.npy'
    np.save(metrics_path, metrics)

    le = LabelEncoder()
    le.fit(y)
    joblib.dump(le, 'trainer/label_encoder.pkl')

    print(f"\nМодель сохранена в {model_path}")
    print(f"Параметры сохранены в {params_path}")
    print(f"Метрики сохранены в {metrics_path}")

    names = load_names()

    y_pred_best = []
    for face in X_test:
        label, _ = best_model.predict(face)
        y_pred_best.append(label)

    cm = confusion_matrix(y_test, y_pred_best)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('reports/confusion_matrix.png', dpi=100, bbox_inches='tight')
    plt.close()

    with open('reports/training_report.txt', 'w', encoding='utf-8') as f:
        f.write("ОТЧЕТ ОБ ОБУЧЕНИИ МОДЕЛИ РАСПОЗНАВАНИЯ ЛИЦ\n")
        f.write("="*50 + "\n\n")
        f.write(f"Дата обучения: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Количество классов: {len(set(labels))}\n")
        f.write(f"Общее количество изображений: {len(faces)}\n")
        f.write(f"Обучающая выборка: {len(X_train)}\n")
        f.write(f"Тестовая выборка: {len(X_test)}\n\n")

        f.write("РАСПРЕДЕЛЕНИЕ ПО КЛАССАМ:\n")
        for label_id, count in label_counter.items():
            name = names.get(label_id, f"ID:{label_id}")
            f.write(f"  {name} (ID:{label_id}): {count} изображений\n")

        f.write("\nРЕЗУЛЬТАТЫ МОДЕЛЕЙ:\n")
        for i, res in enumerate(results):
            f.write(f"\n  Модель {i+1}:\n")
            f.write(f"    Параметры: {res['params']}\n")
            f.write(f"    Accuracy:  {res['accuracy']:.3f}\n")
            f.write(f"    Precision: {res['precision']:.3f}\n")
            f.write(f"    Recall:    {res['recall']:.3f}\n")
            f.write(f"    F1-score:  {res['f1']:.3f}\n")

        f.write(f"\nЛУЧШАЯ МОДЕЛЬ:\n")
        f.write(f"  Параметры: {best_params}\n")
        f.write(f"  Accuracy:  {best_accuracy:.3f}\n")

        if best_accuracy >= MIN_ACCURACY:
            f.write(f"\nМодель принята (accuracy >= {MIN_ACCURACY})\n")
        else:
            f.write(f"\nМодель отклонена (accuracy < {MIN_ACCURACY})\n")

    print(f"Отчет сохранен в reports/training_report.txt")
    print(f"Матрица ошибок сохранена в reports/confusion_matrix.png")

else:
    print(f"\nМодель не сохранена (accuracy {best_accuracy:.3f} < {MIN_ACCURACY})")
    print("   Рекомендации:")
    print("   1. Соберите больше данных")
    print("   2. Увеличьте разнообразие изображений")
    print("   3. Проверьте качество исходных изображений")

print("\n" + "="*50)
print(f'Precision: {prec:.3f}')
print(f'Recall   : {rec:.3f}')
print(f'F1-score : {f1:.3f}')

MIN_ACC = 0.85
if best_accuracy >= MIN_ACC:
    os.makedirs('trainer', exist_ok=True)
    recognizer.write('trainer/trainer.yml')
    print('Модель сохранена')
else:
    print('Модель не сохранена (низкое качество)')
