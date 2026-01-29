import os
import cv2
import numpy as np
import joblib
import sys

sys.path.append(os.path.abspath("."))


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

# SVM configuration finalized from experimentation
SVM_KERNEL = "rbf"
SVM_C = 1.0
SVM_GAMMA = "scale"


def load_dataset():
    X, y = [], []

    for category, label in CATEGORIES.items():
        folder = os.path.join(DATASET_DIR, category)

        if not os.path.exists(folder):
            raise FileNotFoundError(f"Dataset folder not found: {folder}")

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
    print("[INFO] Loading dataset and extracting CNN features...")
    X, y = load_dataset()

    print(f"[INFO] Total samples       : {len(X)}")
    print(f"[INFO] Feature dimension  : {X.shape[1]}")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel=SVM_KERNEL,
            C=SVM_C,
            gamma=SVM_GAMMA
        ))
    ])

    print("[INFO] Training SVM classifier...")
    model.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"[INFO] Trained model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_svm()
