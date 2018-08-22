import cv2
import sys
import time

from detector import FaceDetector
from utils import annotate_image

YOLO_MODELS_DIR = "/Users/ivanitz/Projects/yolo-face-artifacts/run8/models/"
CORRECTOR_MODELS_DIR = "/Users/ivanitz/Projects/fine-tuned-face/models/"


def run(img_path):
    cascPath = "./env/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )


    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    img_path = sys.argv[1]
    run(img_path)
