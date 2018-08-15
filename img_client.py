import cv2
import sys
import time

from detector import FaceDetector
from utils import annotate_image

YOLO_MODELS_DIR = "/Users/ivanitz/Projects/yolo-face-artifacts/run6/models/"
CORRECTOR_MODELS_DIR = "/Users/ivanitz/Projects/fine-tuned-face/models/"


def run(img_path):
    face_detector = FaceDetector(YOLO_MODELS_DIR, CORRECTOR_MODELS_DIR)

    img = cv2.imread(img_path)

    now = time.time()
    bboxes = face_detector.predict(img)
    print("FPS: {:0.2f}".format(1 / (time.time() - now)))
    ann_img = annotate_image(img, bboxes)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image',ann_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    img_path = sys.argv[1]
    run(img_path)
