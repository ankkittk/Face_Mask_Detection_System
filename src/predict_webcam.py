import os
import sys
import cv2
import numpy as np
import joblib

# Ensure project root is in path
sys.path.append(os.path.abspath("."))

from src.feature_extractor import extract_features

MODEL_PATH = "models/svm_model.pkl"


def predict_webcam():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Trained SVM model not found. Train the model first.")

    model = joblib.load(MODEL_PATH)
    print("[INFO] Loaded trained SVM model")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    print("[INFO] Webcam started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Center crop (since training data is face-cropped)
        h, w, _ = frame.shape
        size = int(min(h, w) * 0.6)
        x1 = w // 2 - size // 2
        y1 = h // 2 - size // 2
        x2 = x1 + size
        y2 = y1 + size

        face_roi = frame[y1:y2, x1:x2]

        try:
            features = extract_features(face_roi)
            pred = model.predict([features])[0]

            label = "MASK" if pred == 1 else "NO MASK"
            color = (0, 255, 0) if pred == 1 else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        except Exception:
            pass

        cv2.imshow("Face Mask Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    predict_webcam()
