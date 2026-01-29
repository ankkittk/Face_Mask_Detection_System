import os
import cv2
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.feature_extractor import extract_features

DATASET_DIR = "dataset"
CATEGORIES = {
    "with_mask": 1,
    "without_mask": 0
}

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "svm_model.pkl")


def load_dataset():
    X, y = [], []

    for category, label in CATEGORIES.items():
        folder = os.path.join(DATASET_DIR, category)

        for fname in os.listdir(folder):
            img_path = os.path.join(folder, fname)
            img = cv2.imread(img_path)

            if img is None:
                continue

            features = extract_features(img)
            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)


def train_svm():
    print("[INFO] Loading dataset and extracting features...")
    X, y = load_dataset()

    print(f"[INFO] Total samples: {len(X)}")
    print(f"[INFO] Feature vector length: {X.shape[1]}")

    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale"))
    ])

    print("[INFO] Training SVM...")
    svm_pipeline.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(svm_pipeline, MODEL_PATH)

    print(f"[INFO] Model saved at: {MODEL_PATH}")


if __name__ == "__main__":
    train_svm()
