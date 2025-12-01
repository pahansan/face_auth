import os

import cv2

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
faceNet = cv2.dnn.readNet(faceModel, faceProto)

recognizer = cv2.face.LBPHFaceRecognizer_create()

path = os.path.dirname(os.path.abspath(__file__))
trainer_path = os.path.join(path, 'trainer', 'trainer.yml')
trainer_path = 'trainer/trainer.yml'

if not os.path.exists(trainer_path):
    print(f"❌ Ошибка: Файл {trainer_path} не найден")
    exit()

recognizer.read(trainer_path)
print(f"✅ Модель успешно загружена")

names = {
    1: "Pasha",
    2: "Nikita",
}

CONFIDENCE_THRESHOLD = 70
PADDING = 20

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

video = cv2.VideoCapture(0)

if not video.isOpened():
    print("❌ Не удалось открыть камеру")
    exit()

print("📹 Камера запущена. Нажмите 'Q' для выхода")

while True:
    hasFrame, frame = video.read()
    if not hasFrame:
        break

    resultImg, faceBoxes = highlightFace(faceNet, frame, conf_threshold=0.2)

    if not faceBoxes:
        cv2.putText(resultImg, "Лица не обнаружены", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        for faceBox in faceBoxes:
            x1, y1, x2, y2 = faceBox

            x1_pad = max(0, x1 - PADDING)
            y1_pad = max(0, y1 - PADDING)
            x2_pad = min(frame.shape[1], x2 + PADDING)
            y2_pad = min(frame.shape[0], y2 + PADDING)

            face_roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            if face_roi.size == 0:
                continue

            gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

            label_id, confidence = recognizer.predict(gray_roi)

            if confidence < CONFIDENCE_THRESHOLD:
                name = names.get(label_id, f"ID:{label_id}")
                text = f"{name} ({int(confidence)})"
                color = (0, 255, 0)  # Зелёный
            else:
                text = "Unknown"
                color = (0, 0, 255)  # Красный

            cv2.rectangle(resultImg, (x1, y1), (x2, y2), color, 2)
            cv2.putText(resultImg, text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Face detection and recognition", resultImg)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
print("👋 Программа завершена")
