# coding=utf-8

import os
import cv2
import sys
import argparse
from detector.model import p_net, r_net, o_net
from detector.mtcnn_detector import MtCnnDetector
from config import *
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def test(args):
    detectors = [None, None, None]
    model_path = ['models/p_net/p_net_30', 'models/r_net/r_net_22', 'models/o_net/o_net_22']  # must follow with /
    gray_flag = args.gray_input
    channel = 1 if gray_flag else 3
    if args.input_size == o_net_size:
        detectors[0] = p_net(channel)
        detectors[0].load_weights(model_path[0])
        detectors[1] = r_net(channel)
        detectors[1].load_weights(model_path[1])
        detectors[2] = o_net(channel)
        detectors[2].load_weights(model_path[2])
    elif args.input_size == r_net_size:
        detectors[0] = p_net(channel)
        detectors[0].load_weights(model_path[0])
        detectors[1] = r_net(channel)
        detectors[1].load_weights(model_path[1])
    elif args.input_size == p_net_size:
        detectors[0] = p_net(channel)
        detectors[0].load_weights(model_path[0])
    else:
        print('wrong input size!!')
        return

    mtcnn = MtCnnDetector(detectors, min_face, p_net_stride, face_thresholds)
    input_path = './test_pictures/images'
    output_path = './test_pictures/results/'
    for item in os.listdir(input_path):
        img_path = os.path.join(input_path, item)
        img = cv2.imread(img_path)
        img_proc = img
        if gray_flag:
            img_proc = cv2.cvtColor(img_proc, cv2.COLOR_RGB2GRAY)
        boxes_c, landmarks = mtcnn.detect(img_proc)
        # print('box number = {}, they are {}'.format(boxes_c.shape[0], boxes_c))
        for i in range(boxes_c.shape[0]):
            bbox = boxes_c[i, :4]
            score = boxes_c[i, 4]
            draw_box = [int(x) for x in bbox]
            cv2.rectangle(img, (draw_box[0], draw_box[1]), (draw_box[2], draw_box[3]),
                          (255, 0, 0), 1)
            cv2.putText(img, '{:.2f}'.format(score), (draw_box[0], draw_box[1]-2),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
        for i in range(landmarks.shape[0]):
            for j in range(len(landmarks[i])//2):
                cv2.circle(img, (int(landmarks[i][2*j]), int(landmarks[i][2*j+1])),
                           2, (0, 0, 255))
        cv2.imshow('img', img)
        k = cv2.waitKey(0) & 0xff
        if k == 27:
            cv2.imwrite(output_path+item, img)
    cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_size', type=int,
                        help='The input size for specific net')
    parser.add_argument('--gray_input', type=bool, default=True)
    return parser.parse_args(argv)


if __name__ == '__main__':
    test(parse_arguments(sys.argv[1:]))

