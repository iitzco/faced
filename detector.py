import tensorflow as tf
import cv2
import numpy as np

from const import *


class FaceDetector(object):

    def __init__(self, yolo_model, fine_tune_model):
        self.load_model(yolo_model)
        self.load_aux_vars()

        self.face_corrector = FaceCorrector(fine_tune_model)


    def load_aux_vars(self):
        cols = np.zeros(shape=[1, YOLO_TARGET])
        for i in range(1, YOLO_TARGET):
            cols = np.concatenate((cols, np.full((1, YOLO_TARGET), i)), axis=0)

        self.cols = cols
        self.rows = cols.T

    def load_model(self, model_dir):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session()

            ckpt_path = tf.train.latest_checkpoint(model_dir)
            saver = tf.train.import_meta_graph('{}.meta'.format(ckpt_path))
            saver.restore(self.sess, ckpt_path)

            self.img = tf.get_default_graph().get_tensor_by_name("img:0")
            self.training = tf.get_default_graph().get_tensor_by_name("training:0")
            self.prob = tf.get_default_graph().get_tensor_by_name("prob:0")
            self.x_center = tf.get_default_graph().get_tensor_by_name("x_center:0")
            self.y_center = tf.get_default_graph().get_tensor_by_name("y_center:0")
            self.w = tf.get_default_graph().get_tensor_by_name("w:0")
            self.h = tf.get_default_graph().get_tensor_by_name("h:0")


    def predict(self, frame, thresh=0.8):
        input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (YOLO_SIZE, YOLO_SIZE)) / 255.
        input_img = np.reshape(input_img, [1, YOLO_SIZE, YOLO_SIZE, 3])

        pred = self.sess.run([self.prob, self.x_center, self.y_center, self.w, self.h], feed_dict={self.training: False, self.img: input_img})

        bboxes = self._absolute_bboxes(pred, frame, thresh)
        bboxes_with_status = self._nonmax_supression(bboxes)
        bboxes_with_status = self._correct(frame, bboxes_with_status)

        return bboxes_with_status

    def _absolute_bboxes(self, pred, frame, thresh):
        img_h, img_w, _ = frame.shape
        p, x, y, w, h = pred

        mask = p > thresh

        x += self.cols
        y += self.rows

        p, x, y, w, h = p[mask], x[mask], y[mask], w[mask], h[mask]

        ret = []

        for j in range(x.shape[0]):
            xc, yc = int((x[j]/YOLO_TARGET)*img_w), int((y[j]/YOLO_TARGET)*img_h)
            wi, he = int(w[j]*img_w), int(h[j]*img_h)
            ret.append((xc, yc, wi, he, p[j]))

        return ret

    def _iou(bbox1, bbox2):
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

        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou

    def _nonmax_supression(self, bboxes, thresh=0.2):
        N = len(bboxes)
        status = [SUPPRESSED] * N
        for i in range(N):
            if status[i] == NON_SUPPRESSED:
                continue

            curr_max_p = bboxes[i][-1]
            curr_max_index = i

            for j in range(i+1, N):
                if status[j] == NON_SUPPRESSED:
                    continue

                metric = FaceDetector._iou(bboxes[i], bboxes[j])
                if metric > thresh and bboxes[j][-1] > curr_max_p:
                    curr_max_p = bboxes[j][-1]
                    curr_max_index = j

            status[curr_max_index] = NON_SUPPRESSED

        return zip(bboxes, status)

    def _correct(self, frame, bboxes_with_status):
        ret = list(bboxes_with_status)
        maxs = [b for b in ret if b[-1] == NON_SUPPRESSED]
        N = len(maxs)

        img_h, img_w, _ = frame.shape
        for i in range(N):
            x, y, w, h, p = maxs[i][0]

            # Add margin
            xmin = int(max(0, x - w/2 - 0.5*w))
            xmax = int(min(img_w, x + w/2 + 0.5*w))
            ymin = int(max(0, y - h/2 - 0.5*h))
            ymax = int(min(img_h, y + h/2 + 0.5*h))

            face = frame[ymin:ymax, xmin:xmax, :]
            x, y, w, h = self.face_corrector.predict(face)

            ret.append(((x + xmin, y + ymin, w, h, p), CORRECTED))

        return ret


class FaceCorrector(object):

    def __init__(self, model_dir):
        self.load_model(model_dir)

    def load_model(self, model_dir):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session()

            ckpt_path = tf.train.latest_checkpoint(model_dir)
            saver = tf.train.import_meta_graph('{}.meta'.format(ckpt_path))
            saver.restore(self.sess, ckpt_path)

            self.img = tf.get_default_graph().get_tensor_by_name("img:0")
            self.training = tf.get_default_graph().get_tensor_by_name("training:0")
            self.x = tf.get_default_graph().get_tensor_by_name("X:0")
            self.y = tf.get_default_graph().get_tensor_by_name("Y:0")
            self.w = tf.get_default_graph().get_tensor_by_name("W:0")
            self.h = tf.get_default_graph().get_tensor_by_name("H:0")

    def predict(self, frame):
        # Preprocess
        input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (CORRECTOR_SIZE, CORRECTOR_SIZE)) / 255.
        input_img = np.reshape(input_img, [1, CORRECTOR_SIZE, CORRECTOR_SIZE, 3])

        x, y, w, h = self.sess.run([self.x, self.y, self.w, self.h], feed_dict={self.training: False, self.img: input_img})

        img_h, img_w, _ = frame.shape

        x = int(x*img_w)
        w = int(w*img_w)

        y = int(y*img_h)
        h = int(h*img_h)
        
        return x, y, w, h

