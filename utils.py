import cv2
from const import SUPPRESSED, NON_SUPPRESSED, CORRECTED


def annotate_image(frame, bboxes):
    ret = frame[:]

    img_h, img_w, _ = frame.shape

    for bbox, m in bboxes:
        x, y, w, h, p = bbox

        # if m == SUPPRESSED:
        #     cv2.rectangle(ret, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 255, 0), 3)
        # elif m == NON_SUPPRESSED:
        #     cv2.rectangle(ret, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (255, 0, 0), 3)
        # elif m == CORRECTED:
        #     cv2.rectangle(ret, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 0, 255), 3)
        if m == CORRECTED:
            cv2.rectangle(ret, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 0, 255), 3)

    return ret
