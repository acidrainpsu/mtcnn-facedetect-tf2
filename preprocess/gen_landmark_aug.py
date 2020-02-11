# coding=utf-8

import os
import sys
sys.path.append('../')
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from utils import iou
from bbox_utils import get_data_from_txt, BBox
from config import p_net_size, r_net_size, o_net_size

data_dir = '../data'


def main(args):
    """process landmark data
    """
    size = args.input_size
    augment = args.augment
    if size == p_net_size:
        net = 'p_net'
    elif size == r_net_size:
        net = 'r_net'
    elif size == o_net_size:
        net = 'o_net'
    else:
        net = None

    image_idx = 0
    output_dir = os.path.join(data_dir, net)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    dst_dir = os.path.join(output_dir, 'train_%s_landmark_aug' % net)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    # face label file
    face_txt = os.path.join(data_dir, 'trainImageList.txt')
    f = open(os.path.join(output_dir, 'landmark_%s_aug.txt' % net), 'w')
    data = get_data_from_txt(face_txt, data_dir)
    index = 0
    for image_path, box, landmarks in tqdm(data):
        # store multiple face images and landmarks from one image
        face_list = []
        landmarks_list = []
        image = cv2.imread(image_path)
        image_height, image_width, image_channel = image.shape
        face_box = np.array([box.left, box.top, box.right, box.bottom])
        face_image = image[box.top:box.bottom + 1, box.left:box.right + 1]
        face_image = cv2.resize(face_image, (size, size))
        face_landmarks = np.zeros((5, 2))
        for index, lm in enumerate(landmarks):
            face_landmarks[index] = box.project(landmarks[index])
        face_list.append(face_image)
        landmarks_list.append(face_landmarks.reshape(10))
        if augment:
            face_landmarks = np.zeros((5, 2))
            # transform image
            index += 1
            xl, yl, xr, yr = face_box
            w, h = box.w, box.h
            if max(w, h) < 40 or min(w, h) < 5 or xl < 0 or yl < 0:
                continue
            for i in range(10):
                # random size
                crop_size = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
                # random offset
                delta_x = np.random.randint(-w * 0.2, w * 0.2)
                delta_y = np.random.randint(-h * 0.2, h * 0.2)
                # top-left location of cropped box
                xl_crop = int(max(xl + w / 2 - crop_size / 2 + delta_x, 0))
                yl_crop = int(max(yl + h / 2 - crop_size / 2 + delta_y, 0))
                xr_crop = xl_crop + crop_size
                yr_crop = yl_crop + crop_size
                if xr_crop > image_width or yr_crop > image_height:
                    continue
                cropped_box = np.array([xl_crop, yl_crop, xr_crop, yr_crop])
                iou_value = iou(cropped_box, np.expand_dims(face_box, 0))
                # only keep pos image
                if iou_value > 0.65:
                    cropped_image = image[yl_crop:yr_crop + 1, xl_crop:xr_crop + 1]
                    resized_image = cv2.resize(cropped_image, (size, size), interpolation=cv2.INTER_LINEAR)
                    face_list.append(resized_image)
                    # landmark relative offset
                    for index, lm in enumerate(landmarks):
                        face_landmarks[index] = ((lm[0] - xl_crop) / crop_size, (lm[1] - yl_crop) / crop_size)
                    landmarks_list.append(face_landmarks.reshape(10))
                    cropped_bbox = BBox(cropped_box)
                    # mirror
                    if np.random.choice([0, 1]) > 0:
                        face_flipped, landmark_flipped = flip(resized_image, face_landmarks)
                        face_list.append(face_flipped)
                        landmarks_list.append(landmark_flipped.reshape(10))
                    # rotate clockwise
                    if np.random.choice([0, 1]) > 0:
                        face_rotated_by_alpha, landmark_rotated = \
                            rotate(image, cropped_bbox, cropped_bbox.back_project_landmarks(face_landmarks), 5)
                        # relative offset
                        landmark_rotated = cropped_bbox.project_landmarks(landmark_rotated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size),
                                                           interpolation=cv2.INTER_LINEAR)
                        face_list.append(face_rotated_by_alpha)
                        landmarks_list.append(landmark_rotated.reshape(10))
                        # flip along y axis
                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                        face_list.append(face_flipped)
                        landmarks_list.append(landmark_flipped.reshape(10))
                    # rotate counterclockwise
                    if np.random.choice([0, 1]) > 0:
                        face_rotated_by_alpha, landmark_rotated = \
                            rotate(image, cropped_bbox, cropped_bbox.back_project_landmarks(face_landmarks), -5)
                        # relative offset
                        landmark_rotated = cropped_bbox.project_landmarks(landmark_rotated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size),
                                                           interpolation=cv2.INTER_LINEAR)
                        face_list.append(face_rotated_by_alpha)
                        landmarks_list.append(landmark_rotated.reshape(10))
                        # flip along y axis
                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                        face_list.append(face_flipped)
                        landmarks_list.append(landmark_flipped.reshape(10))
        face_list, landmarks_list = np.asarray(face_list), np.asarray(landmarks_list)
        for i in range(len(face_list)):
            # exclude offset out of [0,1]
            if np.sum(np.where(landmarks_list[i] <= 0, 1, 0)) > 0:
                continue
            if np.sum(np.where(landmarks_list[i] >= 1, 1, 0)) > 0:
                continue
            cv2.imwrite(os.path.join(dst_dir, '%d.jpg' % image_idx), face_list[i])
            landmarks_str = list(map(str, list(landmarks_list[i])))
            f.write(os.path.join(
                dst_dir, '%d.jpg' % image_idx) + ' -2 ' + ' '.join(landmarks_str) + '\n')
            image_idx += 1
    f.close()


def flip(face, landmark):
    # mirror
    face_flipped_by_x = cv2.flip(face, 1)
    landmark_ = np.asarray([(1 - x, y) for (x, y) in landmark])
    landmark_[[0, 1]] = landmark_[[1, 0]]
    landmark_[[3, 4]] = landmark_[[4, 3]]
    return face_flipped_by_x, landmark_


def rotate(img, box, landmark, alpha):
    # rotate
    center = ((box.left + box.right) / 2, (box.top + box.bottom) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)
    img_rotated_by_alpha = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))
    landmark_ = np.asarray([(rot_mat[0][0] * x + rot_mat[0][1] * y + rot_mat[0][2],
                             rot_mat[1][0] * x + rot_mat[1][1] * y + rot_mat[1][2]) for (x, y) in landmark])
    face = img_rotated_by_alpha[box.top:box.bottom + 1, box.left:box.right + 1]
    return face, landmark_


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_size', type=int,
                        help='The input size for specific net')
    parser.add_argument('augment', type=bool, default=True, nargs='?',
                        help='whether enable augment')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
