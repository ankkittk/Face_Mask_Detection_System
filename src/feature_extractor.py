import numpy as np
from preprocess import preprocess_image
from cnn_layers import conv2d, relu, maxpool, flatten


KERNELS = [
    np.array([[1, 0, -1],
              [1, 0, -1],
              [1, 0, -1]]),

    np.array([[ 1, 1, 1],
              [ 0, 0, 0],
              [-1,-1,-1]])
]


def extract_features(img):
    x = preprocess_image(img)
    feature_maps = []

    for kernel in KERNELS:
        conv = conv2d(x, kernel)
        act = relu(conv)
        pool = maxpool(act)
        feature_maps.append(pool)

    feature_maps = np.array(feature_maps)
    features = flatten(feature_maps)
    return features


import cv2

img = cv2.imread("dataset/with_mask/with_mask_1.jpg")
feats = extract_features(img)
print("Feature vector length:", feats.shape[0])
