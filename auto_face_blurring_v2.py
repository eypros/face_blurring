#!/usr/bin/python
# tensorflow code has been taken from: https://github.com/yeephycho/tensorflow-face-detection
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101
import argparse
import sys
import time
import numpy as np
import tensorflow as tf
import cv2
import os

sys.path.append("..")

# from utils import label_map_util

SCORE_THRESH = 0.35
ENLARGE_FACTOR = 0.1
__version__ = '0.1'


def get_arguments():
    """
    Parse the arguments
    :return:
    """
    parser = argparse.ArgumentParser(
        prog='Blur faces in video using openCV',
        description='This module allows to manually annotate inside a video file, skip some frames or skip all video',
        epilog="Developed by: George Orfanidis (g.orfanidis@it.gr)")
    parser.add_argument('-v', '--version', action='version', version='%(prog)s {}'.format(__version__))
    parser.add_argument('video_path',
                        help='the path to the video to be processed')
    parser.add_argument("--score-threshold", type=float, default=SCORE_THRESH,
                        help="The score above which bboxes are taken into consideration. "
                             "Default: {}".format(SCORE_THRESH))
    parser.add_argument("--enlarge-factor", type=float, default=ENLARGE_FACTOR,
                        help="The factor by which the actual detected facial bounding boxes are enlarged before "
                             "applying the blurring mask."
                             "Default: {}".format(ENLARGE_FACTOR))

    args = parser.parse_args()

    return args


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def get_bboxes(boxes_det, scores_det, classes_det, image, score_thres=0.5, enlarge_factor=0.1):
    boxes_det = np.squeeze(boxes_det)
    scores_det = np.squeeze(scores_det)
    classes_det = np.squeeze(classes_det)
    h, w, _ = image.shape
    res = np.where(scores_det > score_thres)
    if not res[0].shape[0]:
        boxes_det = np.zeros((0, 4))
        scores_det = np.zeros((0, 1))
        classes_det = np.zeros((0, 1))
        return boxes_det, scores_det, classes_det
    n = np.where(scores_det > score_thres)[0][-1] + 1

    # this creates an array with just enough rows as object with score above the threshold
    # format: absolute x, y, x, y
    boxes_det = np.array([boxes_det[:n, 1] * w, boxes_det[:n, 0] * h, boxes_det[:n, 3] * w, boxes_det[:n, 2] * h]).T
    classes_det = classes_det[:n]
    scores_det = scores_det[:n]

    # enlarge ROI a bit to make the blurring more effective
    for i in range(boxes_det.shape[0]):
        dx = int(enlarge_factor * (boxes_det[i, 2] - boxes_det[i, 0]))
        dy = int(enlarge_factor * (boxes_det[i, 3] - boxes_det[i, 1]))
        boxes_det[i, 0] = int(boxes_det[i, 0] - dx) if int(boxes_det[i, 0] - dx) > 0 else 0
        boxes_det[i, 1] = int(boxes_det[i, 1] - dy) if int(boxes_det[i, 1] - dy) > 0 else 0
        boxes_det[i, 2] = int(boxes_det[i, 2] + dx) if int(boxes_det[i, 2] + dx) < w else w
        boxes_det[i, 3] = int(boxes_det[i, 3] + dy) if int(boxes_det[i, 3] + dy) < h else h

    return boxes_det, scores_det, classes_det


def main():
    args = get_arguments()
    video_input = args.video_path
    score_threshold = args.score_threshold
    enlarge_factor = args.enlarge_factor

    split_name = os.path.splitext(os.path.basename(video_input))
    video_output = os.path.join(os.path.dirname(video_input), '{}_blurred_auto{}'.format(split_name[0], split_name[1]))

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    path_to_model = './model/frozen_inference_graph_face.pb'

    # List of the strings that is used to add correct label for each box.
    # path_to_labels = './model/face_label_map.pbtxt'
    #
    # num_classes = 2

    # label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    # categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
    #                                                             use_display_name=True)
    # category_index = label_map_util.create_category_index(categories)

    cap = cv2.VideoCapture(video_input)
    out = None
    fourcc = cv2.VideoWriter_fourcc(*'MP43')

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_model, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    with detection_graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(graph=detection_graph, config=config) as sess:
            # frame_num = 1490
            start_time = time.time()
            count = 0
            while True:
                # frame_num -= 1
                ret, frame = cap.read()
                if ret == 0:
                    break
                count += 1

                if count % 50 == 0:
                    print('Processing video {}, frame #{}...'.format(video_input, count))
                if out is None:
                    [h, w] = frame.shape[:2]
                    out = cv2.VideoWriter(video_output, fourcc, 25.0, (w, h))

                image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                boxes, scores, classes = get_bboxes(boxes, scores, classes, frame, score_thres=score_threshold,
                                                    enlarge_factor=enlarge_factor)
                for i in range(len(boxes)):
                    x1 = int(boxes[i][0])
                    y1 = int(boxes[i][1])
                    x2 = int(boxes[i][2])
                    y2 = int(boxes[i][3])

                    # if x1 == 0 or h == 0:
                    #     continue

                    sub_face = frame[y1:y2, x1:x2]

                    # apply a gaussian blur on this new recangle image
                    sub_face = cv2.GaussianBlur(sub_face, (23, 23), 30)

                    # merge this blurry rectangle to our final image
                    frame[y1:y2, x1:x2] = sub_face

                out.write(frame)
            elapsed_time = time.time() - start_time
            fps = count/elapsed_time
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print('Inference time cost: {} with fps {:2.4f} for video {}x{}'.format(elapsed_time, fps, w, h))

            cap.release()
            out.release()


if __name__ == '__main__':
    main()