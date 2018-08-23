import tensorflow as tf
import cv2
import numpy as np
import os

from faced.const import YOLO_TARGET, YOLO_SIZE, CORRECTOR_SIZE, MODELS_PATH
from faced.utils import iou


class FaceDetector(object):

    def __init__(self):
        self.load_model(os.path.join(MODELS_PATH, "face_yolo.pb"))
        self.load_aux_vars()

        self.face_corrector = FaceCorrector()

    def load_aux_vars(self):
        cols = np.zeros(shape=[1, YOLO_TARGET])
        for i in range(1, YOLO_TARGET):
            cols = np.concatenate((cols, np.full((1, YOLO_TARGET), i)), axis=0)

        self.cols = cols
        self.rows = cols.T

    def load_model(self, yolo_model, from_pb=True):
        graph = tf.Graph()
        with graph.as_default():
            self.sess = tf.Session()

            if from_pb:
                with tf.gfile.GFile(yolo_model, "rb") as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    tf.import_graph_def(graph_def, name="") # If not, name is appended in op name

            else:
                ckpt_path = tf.train.latest_checkpoint(yolo_model)
                saver = tf.train.import_meta_graph('{}.meta'.format(ckpt_path))
                saver.restore(self.sess, ckpt_path)

            self.img = tf.get_default_graph().get_tensor_by_name("img:0")
            self.training = tf.get_default_graph().get_tensor_by_name("training:0")
            self.prob = tf.get_default_graph().get_tensor_by_name("prob:0")
            self.x_center = tf.get_default_graph().get_tensor_by_name("x_center:0")
            self.y_center = tf.get_default_graph().get_tensor_by_name("y_center:0")
            self.w = tf.get_default_graph().get_tensor_by_name("w:0")
            self.h = tf.get_default_graph().get_tensor_by_name("h:0")

    # Receives RGB numpy array
    def predict(self, frame, thresh=0.85):
        input_img = cv2.resize(frame, (YOLO_SIZE, YOLO_SIZE)) / 255.
        input_img = np.expand_dims(input_img, axis=0)

        pred = self.sess.run([self.prob, self.x_center, self.y_center, self.w, self.h], feed_dict={self.training: False, self.img: input_img})

        bboxes = self._absolute_bboxes(pred, frame, thresh)
        bboxes = self._correct(frame, bboxes)
        bboxes = self._nonmax_supression(bboxes)

        return bboxes

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

    def _nonmax_supression(self, bboxes, thresh=0.2):
        SUPPRESSED = 1
        NON_SUPPRESSED = 2

        N = len(bboxes)
        status = [None] * N
        for i in range(N):
            if status[i] is not None:
                continue

            curr_max_p = bboxes[i][-1]
            curr_max_index = i

            for j in range(i+1, N):
                if status[j] is not None:
                    continue

                metric = iou(bboxes[i], bboxes[j])
                if metric > thresh:
                    if bboxes[j][-1] > curr_max_p:
                        status[curr_max_index] = SUPPRESSED
                        curr_max_p = bboxes[j][-1]
                        curr_max_index = j
                    else:
                        status[j] = SUPPRESSED

            status[curr_max_index] = NON_SUPPRESSED

        return [bboxes[i] for i in range(N) if status[i] == NON_SUPPRESSED]

    def _correct(self, frame, bboxes):
        N = len(bboxes)
        ret = []

        img_h, img_w, _ = frame.shape
        for i in range(N):
            x, y, w, h, p = bboxes[i]

            MARGIN = 0.5
            # Add margin
            xmin = int(max(0, x - w/2 - MARGIN*w))
            xmax = int(min(img_w, x + w/2 + MARGIN*w))
            ymin = int(max(0, y - h/2 - MARGIN*h))
            ymax = int(min(img_h, y + h/2 + MARGIN*h))

            face = frame[ymin:ymax, xmin:xmax, :]
            x, y, w, h = self.face_corrector.predict(face)

            ret.append((x + xmin, y + ymin, w, h, p))

        return ret


class FaceCorrector(object):

    def __init__(self):
        self.load_model(os.path.join(MODELS_PATH, "face_corrector.pb"))

    def load_model(self, corrector_model, from_pb=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session()
            if from_pb:
                with tf.gfile.GFile(corrector_model, "rb") as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    tf.import_graph_def(graph_def, name="") # If not, name is appended in op name

            else:
                ckpt_path = tf.train.latest_checkpoint(corrector_model)
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
