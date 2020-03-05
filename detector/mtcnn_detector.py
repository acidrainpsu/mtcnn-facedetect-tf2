# coding = utf-8

import cv2
import numpy as np
import sys
from tqdm import tqdm
from preprocess.utils import convert_to_square
from config import p_net_size, p_net_stride, r_net_size, o_net_size


def processed_image(image, scale):
    """
    resize and normalize image
    :param image:
    :param scale:
    :return:
    """
    image = image.astype(np.float32)
    h, w, c = image.shape
    # print('image type = {}, h,w,c = {}-{}-{}, dtype = {}'.format(type(image), h, w, c, image.dtype))
    new_h, new_w = int(h * scale), int(w * scale)
    new_dim = (new_w, new_h)
    img_resized = cv2.resize(image, new_dim, interpolation=cv2.INTER_LINEAR)
    if len(img_resized.shape) < 3:
        img_resized = np.expand_dims(img_resized, -1)
    img_resized = (img_resized - 127.5) / 128
    return img_resized


def generate_bbox(cls_predict, bbox_predict, scale, threshold):
    """"""
    stride = p_net_stride  # p net half the size of image approximately
    cell_size = p_net_size
    h_index = np.where(cls_predict > threshold)  # keep results with high confidence
    # print('cls_predict shape = {}'.format(cls_predict.shape))  # shape (h, w)
    # print('bbox_predict shape = {}'.format(bbox_predict.shape))  # shape(h, w, 4)
    # no face
    if h_index[0].size == 0:
        return np.array([])
    # offset
    dxl, dyl, dxr, dyr = [bbox_predict[h_index[0], h_index[1], i]
                          for i in range(4)]
    bbox_predict = np.array([dxl, dyl, dxr, dyr]) # shape (4, ?)
    # print('bbox_predict shape = {}'.format(bbox_predict.shape))
    score = cls_predict[h_index[0], h_index[1]]
    # combine together
    bounding_box = np.vstack([np.round(stride * h_index[1] / scale),
                              np.round(stride * h_index[0] / scale),
                              np.round((stride * h_index[1] + cell_size) / scale),
                              np.round((stride * h_index[0] + cell_size) / scale),
                              score,
                              bbox_predict])
    return bounding_box.T


def py_nms(detect_boxes, thresh):
    """
    non-maximum suppress
    :param detect_boxes: 
    :param thresh: 
    :return: 
    """
    x1 = detect_boxes[:, 0]
    y1 = detect_boxes[:, 1]
    x2 = detect_boxes[:, 2]
    y2 = detect_boxes[:, 3]
    scores = detect_boxes[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 将概率值从大到小排列
    order = scores.argsort()[::-1]
    # print('order = {}, shape = {}'.format(order, order.shape))
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-10)
        # print('ovr shape = {}'.format(ovr.shape))
        # 保留小于阈值的下标，因为order[0]拿出来做比较了，所以inds+1是原来对应的下标
        inds = np.where(ovr <= thresh)  # return a tuple which first element is an array
        inds = inds[0]
        # print('nms inds shape {}'.format(inds.shape))
        order = order[inds + 1]
    return keep


def pad(boxes, img, output_size):
    """
    crop boxes from img, if one box exceeds image boundaries, crop the box with size unchanged, and \
    its valid part filled with image's content, and exceeded part padded with zeros
    :param boxes:
    :param img:
    :param output_size  output size for next net
    :return: cropped boxes
    """
    height, width, channel = img.shape
    num = boxes.shape[0]
    w, h = boxes[:, 2] - boxes[:, 0] + 1, boxes[:, 3] - boxes[:, 1] + 1
    dxl, dyl = np.zeros((num,)), np.zeros((num,))
    dxr, dyr = w.copy() - 1, h.copy() - 1
    xl, yl, xr, yr = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    # find boxes which exceed bottom & right boundaries
    index = np.where(xr >= width)
    # dxr is the valid right location regard to the left side of the box with width w and height h
    # example: w=151 xl=400 xr=550 width=500 dxr=150 new_dxr=99.
    dxr[index] = w[index] + width - 2 - xr[index]
    xr[index] = width - 1

    index = np.where(yr >= height)
    dyr[index] = h[index] + height - 2 - yr[index]
    yr[index] = height - 1

    # dxl is is the valid left location regard to the left side of the box with width w and height h
    index = np.where(xl < 0)
    dxl[index] = 0 - xl[index]
    xl[index] = 0

    index = np.where(yl < 0)
    dyl[index] = 0 - yl[index]
    yl[index] = 0
    dyl, dyr, dxl, dxr, yl, yr, xl, xr, w, h = [x.astype(np.int32)
                                                for x in [dyl, dyr, dxl, dxr, yl, yr, xl, xr, w, h]]
    # crop boxes with zero padding
    cropped_images = []
    for i in range(num):
        if w[i] < 20 or h[i] < 20 or xl[i] >= width or yl[i] >= height:
            continue
        c_img = np.zeros((h[i], w[i], channel), dtype=np.uint8)
        c_img[dyl[i]:dyr[i]+1, dxl[i]:dxr[i]+1, :] = img[yl[i]: yr[i]+1, xl[i]:xr[i]+1, :]
        cropped_img = (cv2.resize(c_img, (output_size, output_size)) - 127.5) / 128
        cropped_img = np.reshape(cropped_img, (output_size, output_size, channel))
        cropped_images.append(cropped_img)
    if len(cropped_images) == 0:
        return None
    else:
        cropped_images = np.asarray(cropped_images)
        return cropped_images


def calibrate_box(boxes, box_offset):
    """

    :param boxes: boxes from p_net
    :param box_offset: box offsets predicted by r_net
    :return: box absolute locations regard to original image
    """
    bbox = boxes.copy()
    w = boxes[:, 2] - boxes[:, 0] + 1
    h = boxes[:, 3] - boxes[:, 1] + 1
    wh = np.hstack([np.expand_dims(w, 1), np.expand_dims(h, 1)])
    wh_wh = np.hstack([wh, wh])
    abs_offset = box_offset * wh_wh
    bbox[:, :4] = bbox[:, :4] + abs_offset
    return bbox


class MtCnnDetector:
    """
    detector composed of p, r, o net
    """
    def __init__(self, detectors, min_face_size=20,
                 stride=2, thresholds=(0.6, 0.7, 0.7),
                 scale_factor=0.79):
        self.p_net_detector = detectors[0]
        self.r_net_detector = detectors[1]
        self.o_net_detector = detectors[2]
        self.min_face_size = min_face_size
        self.stride = stride
        self.thresholds = thresholds
        self.scale_factor = scale_factor

    def detect_face(self, dataset):
        all_boxes = []
        landmarks = []
        for batch in tqdm(dataset):
            if len(batch.shape) < 3:
                batch = np.expand_dims(batch, -1)
            if self.p_net_detector:
                if type(batch) != np.ndarray:
                    batch = batch.numpy()
                boxes, boxes_c = self.detect_p_net(batch)
                if boxes_c is None:
                    all_boxes.append(np.array([]))
                    landmarks.append(np.array([]))
                    continue
                if self.r_net_detector:
                    boxes, boxes_c = self.detect_r_net(batch, boxes_c)
                    if boxes_c is None:
                        all_boxes.append(np.array([]))
                        landmarks.append(np.array([]))
                        continue
                    # if self.o_net_detector:
                    #     boxes, boxes_c, landmark = self.detect_o_net(batch, boxes_c)
                    #     if boxes_c is None:
                    #         all_boxes.append(np.array([]))
                    #         landmarks.append(np.array([]))
                    #         continue
                all_boxes.append(boxes_c)
                landmark = [1]  # fixme ???
                landmarks.append(landmark)
        return all_boxes, landmarks

    def detect_p_net(self, batch):
        # print('data shape = {}'.format(batch.get_shape()))
        current_scale = float(p_net_size) / self.min_face_size
        image = batch
        print("test image shape ", image.shape)
        im_resized = processed_image(image, current_scale)
        current_height, current_width, _ = im_resized.shape
        all_boxes = []
        while min(current_height, current_width) > p_net_size:
            cls_pred, bbox, _ = self.p_net_detector.predict(np.expand_dims(im_resized, axis=0))
            # print("im_resized size = ", im_resized.shape)
            # print("class map size = ", cls_pred.shape)
            # print("predict class last col = ", cls_pred[0, :, -1, 1])
            boxes = generate_bbox(cls_pred[0, :, :, 1], bbox[0],
                                  current_scale, self.thresholds[0])
            current_scale *= self.scale_factor
            im_resized = processed_image(image, current_scale)
            current_height, current_width, _ = im_resized.shape

            if boxes.size == 0:
                continue
            # non maximum suppress
            keep = py_nms(boxes[:, :5], 0.5)
            boxes = boxes[keep]
            all_boxes.append(boxes)
        if len(all_boxes) == 0:
            return None, None
        all_boxes = np.vstack(all_boxes)
        keep = py_nms(all_boxes[:, :5], 0.6)
        all_boxes = all_boxes[keep]
        # calc width and height of boxes
        bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1
        boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                             all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                             all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                             all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                             all_boxes[:, 4]])
        boxes_c = boxes_c.T
        # print('boxes_c shape = {}'.format(boxes_c))
        boxes = all_boxes[:, :5]
        return boxes, boxes_c

    def detect_r_net(self, batch, detects):
        """
        filter boxes through r_net
        :param batch: image
        :param detects: detected box, absolute axis positions
        :return: box with absolute position
        """
        image = batch
        # h, w, c = image.shape
        detects = convert_to_square(detects)
        detects[:, :4] = np.round(detects[:, :4])
        # print('detects shape = {}, detects = {}'.format(detects.shape, detects))
        # adjust box which exceeds image's size
        cropped_images = pad(detects, image, r_net_size)
        if cropped_images is None:
            return None, None
        cls_predict, boxes_predict, _ = self.r_net_detector.predict(cropped_images)
        cls_predict_positive = cls_predict[:, 1]
        # np.where() return a tuple which first element is an array
        keep_idx = np.where(cls_predict_positive > self.thresholds[1])[0]
        if len(keep_idx) > 0:
            boxes = detects[keep_idx]
            boxes[:, 4] = cls_predict_positive[keep_idx]  # update probability
            boxes_predict = boxes_predict[keep_idx]
        else:
            return None, None
        keep = py_nms(boxes, 0.6)
        boxes = boxes[keep]
        boxes_predict = boxes_predict[keep]
        # calibrate box locations according to new predicts by r_net
        boxes_c = calibrate_box(boxes, boxes_predict)
        return boxes, boxes_c

    def detect_o_net(self, batch, detects):
        image = batch
        detects = convert_to_square(detects)
        detects[:, :4] = np.round(detects[:, :4])
        cropped_images = pad(detects, image, o_net_size)
        if cropped_images is None:
            return None, None
        cls_predict, boxes_predict, landmark = self.o_net_detector.predict(cropped_images)
        cls_predict_positive = cls_predict[:, 1]
        keep_idx = np.where(cls_predict_positive > self.thresholds[2])[0]
        if len(keep_idx) > 0:
            boxes = detects[keep_idx]
            boxes[:, 4] = cls_predict_positive[keep_idx]  # update probability
            boxes_predict = boxes_predict[keep_idx]
            landmark = landmark[keep_idx]
        else:
            return None, None

        w = boxes[:, 2] - boxes[:, 0] + 1
        h = boxes[:, 3] - boxes[:, 1] + 1
        landmark[:, 0::2] = (np.tile(w, (5, 1)) * landmark[:, 0::2].T + np.tile(boxes[:, 0], (5, 1)) - 1).T
        landmark[:, 1::2] = (np.tile(h, (5, 1)) * landmark[:, 1::2].T + np.tile(boxes[:, 1], (5, 1)) - 1).T
        boxes_c = calibrate_box(boxes, boxes_predict)
        keep = py_nms(boxes_c, 0.6)
        boxes_c = boxes_c[keep]
        landmark = landmark[keep]
        return boxes_c, landmark

    def detect(self, image):
        """
        for test
        :param image:
        :return:
        """
        if len(image.shape) < 3:
            image = np.expand_dims(image, -1)
        boxes_c, landmark = np.array([]), np.array([])
        if self.p_net_detector:
            boxes, boxes_c = self.detect_p_net(image)
            if boxes_c is None:
                return np.array([]), np.array([])

            if self.r_net_detector:
                boxes, boxes_c = self.detect_r_net(image, boxes_c)
                if boxes_c is None:
                    return np.array([]), np.array([])

                if self.o_net_detector:
                    boxes_c, landmark = self.detect_o_net(image, boxes_c)
                    if boxes_c is None:
                        return np.array([]), np.array([])
        return boxes_c, landmark




