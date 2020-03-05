# coding=utf-8

import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm
from preprocess.utils import iou
from preprocess.bbox_utils import get_data_from_txt, BBox


def gen_landmark(data_dir, size, output_file, output_data_file, gray, augment=True):
    face_txt = os.path.join(data_dir, 'trainImageList.txt')
    f = open(output_file, 'w')
    data = get_data_from_txt(face_txt, data_dir)
    index = 0
    landmark_face_all = []
    for image_path, box, landmarks in tqdm(data):
        # store multiple face images and landmarks from one image
        face_list = []
        landmarks_list = []
        image = cv2.imread(image_path)
        image_height, image_width, image_channel = image.shape
        if gray:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        face_box = np.array([box.left, box.top, box.right, box.bottom])
        face_image = cv2.resize(image[box.top:box.bottom + 1, box.left:box.right + 1],
                                (size, size))
        # projected(relative) landmark positions
        projected_landmarks = np.zeros((5, 2))
        for idx, _ in enumerate(landmarks):
            projected_landmarks[idx] = box.project(landmarks[idx])
        face_list.append(face_image)
        landmarks_list.append(projected_landmarks.reshape(10))
        if augment:
            projected_landmarks = np.zeros((5, 2))
            # transform image
            xl, yl, xr, yr = face_box
            w, h = box.w, box.h
            if max(w, h) < 40 or min(w, h) < 10 or xl < 0 or yl < 0:
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
                    cropped_bbox = BBox(cropped_box)
                    resized_image = cv2.resize(image[yl_crop:yr_crop + 1, xl_crop:xr_crop + 1],
                                               (size, size), interpolation=cv2.INTER_LINEAR)
                    face_list.append(resized_image)
                    # landmark relative offset
                    projected_landmarks = cropped_bbox.project_landmarks(landmarks)
                    landmarks_list.append(projected_landmarks.reshape(10))
                    # mirror
                    if np.random.choice([0, 1]) > 0:
                        face_flipped, landmark_flipped = flip(resized_image, projected_landmarks)
                        face_list.append(face_flipped)
                        landmarks_list.append(landmark_flipped.reshape(10))
                    # rotate clockwise
                    if np.random.choice([0, 1]) > 0:
                        face_rotated_by_alpha, landmark_rotated = \
                            rotate(image, cropped_bbox, landmarks, 5)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size),
                                                           interpolation=cv2.INTER_LINEAR)
                        landmark_rotated = cropped_bbox.project_landmarks(landmark_rotated)
                        face_list.append(face_rotated_by_alpha)
                        landmarks_list.append(landmark_rotated.reshape(10))
                        # mirror: flip along y axis, add one more
                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                        face_list.append(face_flipped)
                        landmarks_list.append(landmark_flipped.reshape(10))
                    # rotate counterclockwise
                    if np.random.choice([0, 1]) > 0:
                        face_rotated_by_alpha, landmark_rotated = \
                            rotate(image, cropped_bbox, landmarks, -5)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size),
                                                           interpolation=cv2.INTER_LINEAR)
                        landmark_rotated = cropped_bbox.project_landmarks(landmark_rotated)
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
            landmark_face_all.append(face_list[i])
            landmarks_str = list(map(str, list(landmarks_list[i])))
            f.write(str(index) + ' -2 ' + ' '.join(landmarks_str) + '\n')
            index += 1
    num = len(landmark_face_all)
    assert num == index
    f.close()
    with open(output_data_file, 'wb') as f_data:
        pickle.dump(landmark_face_all, f_data, 1)
    return num


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
