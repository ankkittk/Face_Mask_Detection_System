import cv2
import numpy as np


def preprocess_image(img, output_size=(64, 64)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, output_size)
    return resized



img = cv2.imread(r"dataset\with_mask\with_mask_1.jpg")
processed = preprocess_image(img)

#cv2.imshow("Processed Face Image", processed)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
