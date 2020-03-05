# coding = utf-8

import numpy as np
import cv2
import tensorflow as tf


def iou(box, boxes):
    """iou value between cropped box and all annotated box
    :parameter:
        box: cropped box, first 4 numbers represent top-left, bottom-right point locations,
            and optional fifth number which is confidence
        boxes: all annotated face box, [n, 4]
    :return
        iou value, [n,]
    """
    if len(boxes) == 0:
        return 0
    # area
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    # areas
    boxes_areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    # overlapped part's top-left, bottom right point
    xl = np.maximum(box[0], boxes[:, 0])
    yl = np.maximum(box[1], boxes[:, 1])
    xr = np.minimum(box[2], boxes[:, 2])
    yr = np.minimum(box[3], boxes[:, 3])

    # overlapped width, height and area
    width = np.maximum(0, xr - xl + 1)
    height = np.maximum(0, yr - yl + 1)
    area = width * height
    return area / (box_area + boxes_areas - area + 1e-10)


def read_annotations(images_dir, label_path):
    """
    read images and bounding boxes
    :param images_dir:
    :param label_path:
    :return:
    """
    data = dict()
    images = []
    bboxes = []
    label_file = open(label_path, 'r')
    while True:
        # 图像地址
        image_path = label_file.readline().strip('\n')
        # print('image path %s' % image_path)
        if not image_path:
            break
        image_path = images_dir + image_path
        images.append(image_path)
        # 人脸数目
        nums = int(label_file.readline().strip('\n'))
        # print('box num %s' % nums)
        bboxes_in_one_image = []
        if nums == 0:
            null = label_file.readline().strip('\n')
            bboxes_in_one_image.append([0, 0, 0, 0])
        else:
            for i in range(int(nums)):
                bbox_info = label_file.readline().strip('\n').split(' ')
                face_box = [float(bbox_info[i]) for i in range(4)]

                xl = face_box[0]
                yl = face_box[1]
                xr = xl + face_box[2]
                yr = yl + face_box[3]

                bboxes_in_one_image.append([xl, yl, xr, yr])

        bboxes.append(bboxes_in_one_image)

    data['images'] = images
    data['bboxes'] = bboxes
    return data


def load_data_to_dataset(data, channel):
    def gen():
        for img_path in data:
            img = cv2.imread(img_path)
            if channel < 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            yield img

    dataset = tf.data.Dataset.from_generator(
        gen,
        tf.int64,
        tf.TensorShape((None, None, channel)))
    return dataset


def convert_to_square(box):
    """将box转换成更大的正方形
    参数：
      box：预测的box,[n,5]
    返回值：
      调整后的正方形box，[n,5]
    """
    square_box = box.copy()
    h = box[:, 3] - box[:, 1] + 1
    w = box[:, 2] - box[:, 0] + 1
    # 找寻正方形最大边长
    max_side = np.maximum(w, h)

    square_box[:, 0] = box[:, 0] + w * 0.5 - max_side * 0.5
    square_box[:, 1] = box[:, 1] + h * 0.5 - max_side * 0.5
    square_box[:, 2] = square_box[:, 0] + max_side - 1
    square_box[:, 3] = square_box[:, 1] + max_side - 1
    return square_box
