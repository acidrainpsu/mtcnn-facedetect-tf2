# coding=utf-8

import os
import numpy as np


def get_data_from_txt(txt, data_path, with_landmark=True):
    """
    get images, boxes, landmarks from txt
    :param txt: txt file
    :param data_path: images directory
    :param with_landmark: with or without landmarks
    :return: list contains (image, box, landmark)
    """
    with open(txt, 'r') as f:
        lines = f.readlines()
    result = []
    for line in lines:
        line = line.strip()
        components = line.split(' ')
        image_path = os.path.join(data_path, components[0]).replace('\\', '/')
        box = (components[1], components[3], components[2], components[4])
        box = [float(_) for _ in box]
        box = list(map(int, box))

        if not with_landmark:
            result.append((image_path, BBox(box)))
            continue
        # five landmarks
        landmark = np.zeros((5, 2))
        for index in range(5):
            point = (float(components[5+2*index]), float(components[5+2*index+1]))
            landmark[index] = point
        result.append((image_path, BBox(box), landmark))
    return result


# bounding box class
class BBox:
    def __init__(self, box):
        self.left = box[0]
        self.top = box[1]
        self.right = box[2]
        self.bottom = box[3]
        self.x = box[0]
        self.y = box[1]
        self.w = box[2] - box[0]
        self.h = box[3] - box[1]

    def project(self, point):
        """
        project point to relative locations with regard to box's top-left
        :param point:
        :return:
        """
        x = (point[0] - self.x) / self.w
        y = (point[1] - self.y) / self.h
        return np.asarray([x, y])

    def back_project(self, point):
        """
        back project point from relative locations, reverse version of project
        :param point:
        :return:
        """
        x = self.x + self.w * point[0]
        y = self.y + self.h * point[1]
        return np.asarray([x, y])

    def project_landmarks(self, landmarks):
        """
        project multiple landmark points
        :param landmarks:
        :return:
        """
        length = len(landmarks)
        ret = np.zeros((length, 2))
        for i in range(length):
            ret[i] = self.project(landmarks[i])
        return ret

    def back_project_landmarks(self, landmarks):
        """
        back project multiple landmark points
        :param landmarks:
        :return:
        """
        length = len(landmarks)
        ret = np.zeros((length, 2))
        for i in range(length):
            ret[i] = self.back_project(landmarks[i])
        return ret
