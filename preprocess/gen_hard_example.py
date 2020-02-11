# coding=utf-8

import sys
sys.path.append('../')
import numpy as np
import argparse
import os
import cv2
import pickle
from tqdm import tqdm
from utils import *
import train.train_config as tc
from train.model import p_net, r_net, o_net
from detector.mtcnn_detector import MtCnnDetector
from config import p_net_size, r_net_size, o_net_size, \
    min_face, p_net_stride, face_thresholds


physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def main(args):
    """
    generate next
    :param args:
    :return:
    """
    size = args.input_size
    # models
    model_path = ['../models/p_net/', '../models/r_net/', '../models/o_net/']
    if size == p_net_size:
        net = 'r_net'
        save_size = r_net_size
    elif size == r_net_size:
        net = 'o_net'
        save_size = o_net_size
    else:
        return
    # images path
    image_dir = '../data/WIDER_train/images/'
    output_dir = '../data/%s' % net
    neg_dir = os.path.join(output_dir, 'negative')
    pos_dir = os.path.join(output_dir, 'positive')
    part_dir = os.path.join(output_dir, 'partial')
    for dir_path in [neg_dir, pos_dir, part_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    detectors = [None, None, None]
    net_p = p_net()
    net_p.load_weights(model_path[0])
    detectors[0] = net_p
    if size == r_net_size:
        net_r = r_net()
        net_r.load_weights(model_path[1])
        detectors[1] = net_r
        print("r_net loaded!")
    wider_face_file = '../data/wider_face_train_bbx_gt.txt'
    data = read_annotations(image_dir, wider_face_file)
    mtcnn = MtCnnDetector(detectors, min_face, p_net_stride, face_thresholds)
    save_dir = '../data'
    save_detects_file = os.path.join(save_dir, net + '_detections.pkl')
    if not os.path.exists(save_detects_file):
        print('loading data to dataset')
        loaded_dataset = load_data_to_dataset(data['images'])
        print('starting to detect')
        detect_result, _ = mtcnn.detect_face(loaded_dataset.take(3000))  # fixme Not only 3000
        print('detect over')
        with open(save_detects_file, 'wb') as f:
            pickle.dump(detect_result, f, 1)
    print('start to generate hard image')
    save_hard_example(save_size, data, neg_dir, pos_dir, part_dir, save_detects_file)


def save_hard_example(save_size, data, neg_dir, pos_dir, part_dir, saved_file):
    """
    crop original image using previous net outputted boxes for next net
    :param save_size:
    :param data:
    :param neg_dir: dir to put neg images
    :param pos_dir:
    :param part_dir:
    :param saved_file:
    :return:
    """
    img_list = data['images']
    gt_boxes_list = data['bboxes']
    num_img = len(img_list)
    if save_size == r_net_size:
        net = 'r_net'
    elif save_size == o_net_size:
        net = 'o_net'
    else:
        return

    neg_label_file, pos_label_file, part_label_file = \
        ["../data/%s/%s_%s.txt" % (net, net, i) for i in ['neg', 'pos', 'part']]
    neg_file, pos_file, part_file = \
        [open(file, 'w') for file in [neg_label_file, pos_label_file, part_label_file]]
    # read detect results
    with open(saved_file, 'rb') as sf:
        detected_box = pickle.load(sf)
    print('num of detected_box is {}, num of images is {}.'.format(
        len(detected_box), num_img))
    neg_idx, pos_idx, part_idx = 0, 0, 0
    proc_idx = 0
    for img_idx, detect_box, gt_box in tqdm(zip(img_list, detected_box, gt_boxes_list)):
        gt_box = np.array(gt_box, dtype=np.float32).reshape(-1, 4)
        proc_idx += 1
        if detect_box.shape[0] == 0:
            continue
        img = cv2.imread(img_idx)
        detect_box = convert_to_square(detect_box)
        detect_box[:, :4] = np.round(detect_box[:, :4])
        neg_num = 0
        # print("proc_idx = {}, gt_box = {}".format(proc_idx, gt_box))
        for box in detect_box:
            xl, yl, xr, yr, _ = box.astype(int)
            width = xr - xl + 1
            height = yr - yl + 1

            # filter too small or exceed boundary
            if width < 20 or height < 20 or xl < 0 or yl < 0 or\
                    xr >= img.shape[1] or yr >= img.shape[0]:
                continue
            iou_value = iou(box, gt_box)
            cropped_img = img[yl:yr+1, xl:xr+1, :]
            resized_img = cv2.resize(cropped_img, (save_size, save_size),
                                     interpolation=cv2.INTER_LINEAR)
            if np.max(iou_value) < 0.3 and neg_num < 60:
                save_file = os.path.join(neg_dir, "%s.jpg" % neg_idx)
                neg_file.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_img)
                neg_idx += 1
                neg_num += 1
            else:
                idx = np.argmax(iou_value)
                corresponding_gt_box = gt_box[idx]
                xl_gt, yl_gt, xr_gt, yr_gt = corresponding_gt_box

                offset_xl = (xl_gt - xl) / float(width)
                offset_yl = (yl_gt - yl) / float(height)
                offset_xr = (xr_gt - xr) / float(width)
                offset_yr = (yr_gt - yr) / float(height)
                if np.max(iou_value) >= 0.65:
                    save_file = os.path.join(pos_dir, '%s.jpg' % pos_idx)
                    pos_file.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' %
                                   (offset_xl, offset_yl, offset_xr, offset_yr))
                    cv2.imwrite(save_file, resized_img)
                    pos_idx += 1
                elif np.max(iou_value) >= 0.4:
                    save_file = os.path.join(part_dir, '%s.jpg' % part_idx)
                    part_file.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' %
                                    (offset_xl, offset_yl, offset_xr, offset_yr))
                    cv2.imwrite(save_file, resized_img)
                    part_idx += 1
        if proc_idx >= min(len(detected_box), num_img):
            break
    neg_file.close()
    part_file.close()
    pos_file.close()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_size', type=int,
                        help='The input size for specific net')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
