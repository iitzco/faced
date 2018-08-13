import cv2
import sys
import time

from detector import FaceDetector
from utils import annotate_image

YOLO_MODELS_DIR = "/Users/ivanitz/Projects/yolo-face-artifacts/run6/models/"
CORRECTOR_MODELS_DIR = "/Users/ivanitz/Projects/fine-tuned-face/models/"


def run(feed):
    face_detector = FaceDetector(YOLO_MODELS_DIR, CORRECTOR_MODELS_DIR)

    if feed is None:
        # From webcam
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(feed)

    now = time.time()
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        now = time.time()
        bboxes = face_detector.predict(frame)
        print("FPS: {:0.2f}".format(1 / (time.time() - now)))
        ann_frame = annotate_image(frame, bboxes)

        # Display the resulting frame
        cv2.imshow('frame', ann_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    feed = None if len(sys.argv) == 1 else sys.argv[1]
    run(feed)
