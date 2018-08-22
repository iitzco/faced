import cv2
import sys
import os

from faced import FaceDetector
from faced.utils import annotate_image


def run(img_path):
    face_detector = FaceDetector()

    img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    bboxes = face_detector.predict(rgb_img)
    ann_img = annotate_image(img, bboxes)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image',ann_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    img_path = sys.argv[1]
    run(img_path)
