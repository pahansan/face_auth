import os
import time
from collections import deque

import cv2
import joblib
import numpy as np

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
faceNet = cv2.dnn.readNet(faceModel, faceProto)

recognizer = cv2.face.LBPHFaceRecognizer_create()

model_files = [
    'trainer/trainer.yml',
    'trainer/model_params.npy',
    'trainer/label_encoder.pkl'
]

for file in model_files:
    if not os.path.exists(file):
        print(f"File not found: {file}")
        exit()

recognizer.read('trainer/trainer.yml')
model_params = np.load('trainer/model_params.npy', allow_pickle=True).item()
label_encoder = joblib.load('trainer/label_encoder.pkl')

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

names = load_names()

CONFIDENCE_THRESHOLD = 70
DETECTION_CONFIDENCE = 0.3
PADDING = 20
FACE_SIZE = (200, 200)

PREDICTION_HISTORY_LENGTH = 1
MIN_CONSECUTIVE_FRAMES = 5
STABILITY_THRESHOLD = 0.7
MAX_TRACKER_AGE = 0.5

class FaceTracker:
    def __init__(self, max_history=PREDICTION_HISTORY_LENGTH):
        self.trackers = {}
        self.next_id = 0
        self.max_history = max_history

    def update(self, face_boxes):
        current_time = time.time()
        trackers_to_check = {
            id: t for id, t in self.trackers.items()
            if current_time - t.get('last_seen', 0) < MAX_TRACKER_AGE
        }

        updated_trackers = {}

        unassigned_trackers = dict(trackers_to_check)
        unassigned_boxes_indices = list(range(len(face_boxes)))

        for i in unassigned_boxes_indices:
            box = face_boxes[i]
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            best_id = None
            min_distance = float('inf')

            for tracker_id, tracker in unassigned_trackers.items():
                last_box = tracker['positions'][-1]
                last_center_x = (last_box[0] + last_box[2]) / 2
                last_center_y = (last_box[1] + last_box[3]) / 2

                distance = np.sqrt((center_x - last_center_x)**2 +
                                  (center_y - last_center_y)**2)

                max_move_distance = 100
                if distance < min_distance and distance < max_move_distance:
                    min_distance = distance
                    best_id = tracker_id

            if best_id is not None:
                tracker = unassigned_trackers[best_id]
                tracker['positions'].append(box)
                if len(tracker['positions']) > self.max_history:
                    tracker['positions'].popleft()

                tracker['last_seen'] = current_time

                updated_trackers[best_id] = tracker
                del unassigned_trackers[best_id]
            else:
                tracker = {
                    'positions': deque([box], maxlen=self.max_history),
                    'predictions': deque(maxlen=self.max_history),
                    'last_seen': current_time,
                    'stable_id': None,
                    'stable_confidence': 0
                }
                updated_trackers[self.next_id] = tracker
                self.next_id += 1

        self.trackers = updated_trackers
        return self.trackers

    def add_prediction(self, tracker_id, prediction, confidence):
        if tracker_id not in self.trackers:
            return

        self.trackers[tracker_id]['predictions'].append((prediction, confidence))
        self.trackers[tracker_id]['last_seen'] = time.time()

        if len(self.trackers[tracker_id]['predictions']) < MIN_CONSECUTIVE_FRAMES:
            return

        predictions = [p[0] for p in self.trackers[tracker_id]['predictions']]
        confidences = [p[1] for p in self.trackers[tracker_id]['predictions']]

        predictions = np.asarray(predictions, dtype=np.int32)
        confidences = np.asarray(confidences, dtype=np.float64)

        valid = (predictions >= 0) & np.isfinite(confidences)
        if not np.any(valid):
            return
        predictions = predictions[valid]
        confidences = confidences[valid]

        unique, counts = np.unique(predictions, return_counts=True)
        max_count = np.max(counts)

        if max_count / len(predictions) >= STABILITY_THRESHOLD:
            most_common = unique[np.argmax(counts)]

            mask = [p == most_common for p in predictions]
            selected = [confidences[i] for i, m in enumerate(mask) if m]
            if not selected:
                return

            avg_confidence = float(np.mean(selected))

            self.trackers[tracker_id]['stable_id'] = most_common
            self.trackers[tracker_id]['stable_confidence'] = avg_confidence


def detect_faces_dnn(frame, conf_threshold=DETECTION_CONFIDENCE):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 [104, 117, 123], True, False)
    faceNet.setInput(blob)
    detections = faceNet.forward()

    face_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            face_boxes.append([x1, y1, x2, y2])

    return face_boxes

def preprocess_face(face_img, target_size=FACE_SIZE):
    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img

    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    resized = cv2.resize(gray, target_size)

    return resized

def draw_face_info(frame, box, label_id, confidence, tracker_id=None):
    x1, y1, x2, y2 = box

    if confidence < CONFIDENCE_THRESHOLD:
        name = names.get(label_id, f"ID:{label_id}")
        text = f"{name} ({int(confidence)})"
        color = (0, 255, 0)
    else:
        text = "Unknown"
        color = (0, 0, 255)

    if tracker_id is not None:
        text = f"{text}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )

    cv2.rectangle(frame,
                  (x1, y1 - text_height - 10),
                  (x1 + text_width, y1),
                  color, -1)

    cv2.putText(frame, text, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame

video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Camera error")
    exit()

face_tracker = FaceTracker()

frame_count = 0
fps = 0
start_time = time.time()

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    frame_count += 1

    face_boxes = detect_faces_dnn(frame, DETECTION_CONFIDENCE)

    trackers = face_tracker.update(face_boxes)

    for tracker_id, tracker in trackers.items():
        x1, y1, x2, y2 = tracker['positions'][-1]

        x1_pad = max(0, x1 - PADDING)
        y1_pad = max(0, y1 - PADDING)
        x2_pad = min(frame.shape[1], x2 + PADDING)
        y2_pad = min(frame.shape[0], y2 + PADDING)

        face_roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]
        if face_roi.size == 0:
            continue

        processed_face = preprocess_face(face_roi)

        label_id, confidence = recognizer.predict(processed_face)

        face_tracker.add_prediction(tracker_id, label_id, confidence)

        if tracker['stable_id'] is not None:
            label_id = tracker['stable_id']
            confidence = tracker['stable_confidence']

        frame = draw_face_info(frame, (x1, y1, x2, y2),
                              label_id, confidence, tracker_id)

    current_time = time.time()
    if current_time - start_time >= 1.0:
        fps = frame_count / (current_time - start_time)
        frame_count = 0
        start_time = current_time

    info_text = [
        f"FPS: {fps:.1f}",
        f"Faces: {len(trackers)}",
        f"Threshold: {CONFIDENCE_THRESHOLD}"
    ]

    for i, text in enumerate(info_text):
        cv2.putText(frame, text, (10, 30 + i*25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Face Recognition System", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        face_tracker = FaceTracker()
    elif key == ord('+'):
        CONFIDENCE_THRESHOLD = min(100, CONFIDENCE_THRESHOLD + 5)
    elif key == ord('-'):
        CONFIDENCE_THRESHOLD = max(0, CONFIDENCE_THRESHOLD - 5)

video.release()
cv2.destroyAllWindows()
