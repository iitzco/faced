import cv2

def iou(bbox1, bbox2):
    # determine the (x, y)-coordinates of the intersection rectangle
    boxA = bbox1[0] - bbox1[2]/2, bbox1[1] - bbox1[3]/2, bbox1[0] + bbox1[2]/2, bbox1[1] + bbox1[3]/2
    boxB = bbox2[0] - bbox2[2]/2, bbox2[1] - bbox2[3]/2, bbox2[0] + bbox2[2]/2, bbox2[1] + bbox2[3]/2

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    ret = interArea / float(boxAArea + boxBArea - interArea)

    return ret


def annotate_image(frame, bboxes):
    ret = frame[:]

    img_h, img_w, _ = frame.shape

    for x, y, w, h, p in bboxes:
        cv2.rectangle(ret, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 255, 0), 3)

    return ret
