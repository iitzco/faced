import os

OUTPUT_DIR = "/artifacts"
INPUT_DIR = "/storage/widerface"

TRAIN_TF_RECORD_FILE = os.path.join(INPUT_DIR, "train.tfrecord")
VAL_TF_RECORD_FILE = os.path.join(INPUT_DIR, "val.tfrecord")

# YOLO_SIZE = 352
# YOLO_TARGET = 11
YOLO_SIZE = 288
YOLO_TARGET = 9

CORRECTOR_SIZE = 50


SUPPRESSED = 1
NON_SUPPRESSED = 2
CORRECTED = 3
