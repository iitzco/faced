import cv2
import sys
import time

from detector import FaceDetector
from utils import annotate_image

YOLO_MODELS_DIR = "/home/ivanitz/yolo-face/models/"
CORRECTOR_MODELS_DIR = "/home/ivanitz/face-correction/models/"


def run(feed):
    face_detector = FaceDetector(YOLO_MODELS_DIR, CORRECTOR_MODELS_DIR)

    if feed is None:
        # From webcam
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(feed)

    # Get current width of frame
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    # Get current height of frame
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    fps = cap.get(cv2.CAP_PROP_FPS) # float


    # Define the codec and create VideoWriter object
    # fourcc = cv2.Video.CV_FOURCC(*'X264')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter("output.avi",fourcc, fps, (int(width),int(height)))


    now = time.time()
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        now = time.time()
        bboxes = face_detector.predict(frame)
        print("FPS: {:0.2f}".format(1 / (time.time() - now)))
        ann_frame = annotate_image(frame, bboxes)

        out.write(ann_frame)

        # Display the resulting frame
        # cv2.imshow('frame', ann_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    feed = None if len(sys.argv) == 1 else sys.argv[1]
    run(feed)
