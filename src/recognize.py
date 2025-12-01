import os

import cv2

# ========== 1. ЗАГРУЗКА МОДЕЛЕЙ ==========

# DNN-модель для обнаружения лиц
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# LBPH-модель для распознавания (обученная вами)
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Путь к обученной модели
path = os.path.dirname(os.path.abspath(__file__))
trainer_path = os.path.join(path, 'trainer', 'trainer.yml')
trainer_path = 'trainer/trainer.yml'

# Проверяем, существует ли файл модели
if not os.path.exists(trainer_path):
    print(f"❌ Ошибка: Файл {trainer_path} не найден. Сначала запустите скрипт обучения!")
    exit()

# Загружаем обученные данные
recognizer.read(trainer_path)
print(f"✅ Модель успешно загружена")

# ========== 2. НАСТРОЙКИ ==========

# Словарь ID -> Имя (заполните своими данными!)
# Например: {1: "Иван", 2: "Мария"}
names = {
    1: "Pashka",
    2: "User_2",
    3: "User_3",
    # Добавьте сюда ID и имена из вашего набора данных
}

# Порог уверенности (чем МЕНЬШЕ, тем строже. LBPH возвращает "расстояние")
# Оптимально: 50-80. Если слишком много ложных срабатываний - уменьшите
CONFIDENCE_THRESHOLD = 70

# Отступ вокруг лица для лучшего распознавания
PADDING = 20

# ========== 3. ФУНКЦИИ ==========

def highlightFace(net, frame, conf_threshold=0.7):
    """Обнаруживает лица с помощью DNN"""
    frameOpencvDnn = frame.copy()
    frameHeight, frameWidth = frameOpencvDnn.shape[:2]

    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()

    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])

    return frameOpencvDnn, faceBoxes

# ========== 4. ЗАПУСК ВИДЕО ==========

video = cv2.VideoCapture(0)

if not video.isOpened():
    print("❌ Не удалось открыть камеру")
    exit()

print("📹 Камера запущена. Нажмите 'Q' для выхода")

while True:
    hasFrame, frame = video.read()
    if not hasFrame:
        break

    # Обнаруживаем лица
    resultImg, faceBoxes = highlightFace(faceNet, frame, conf_threshold=0.2)

    if not faceBoxes:
        cv2.putText(resultImg, "Лица не обнаружены", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        # Распознаём каждое найденное лицо
        for faceBox in faceBoxes:
            x1, y1, x2, y2 = faceBox

            # Добавляем отступы
            x1_pad = max(0, x1 - PADDING)
            y1_pad = max(0, y1 - PADDING)
            x2_pad = min(frame.shape[1], x2 + PADDING)
            y2_pad = min(frame.shape[0], y2 + PADDING)

            # Вырезаем лицо и конвертируем в grayscale
            face_roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            if face_roi.size == 0:
                continue

            gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

            # Распознавание
            label_id, confidence = recognizer.predict(gray_roi)

            # Формируем текст
            if confidence < CONFIDENCE_THRESHOLD:
                name = names.get(label_id, f"ID:{label_id}")
                text = f"{name} ({int(confidence)})"
                color = (0, 255, 0)  # Зелёный
            else:
                text = "Unknown"
                color = (0, 0, 255)  # Красный

            # Рисуем прямоугольник и текст
            cv2.rectangle(resultImg, (x1, y1), (x2, y2), color, 2)
            cv2.putText(resultImg, text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Face detection and recognition", resultImg)

    # Выход по Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ========== 5. ОЧИСТКА ==========
video.release()
cv2.destroyAllWindows()
print("👋 Программа завершена")
