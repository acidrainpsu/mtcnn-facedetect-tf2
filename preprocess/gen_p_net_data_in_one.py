# coding=utf-8

import os
import cv2
import numpy as np
from tqdm import tqdm
from preprocess.utils import iou


def is_valid_box(box):
    xl, yl, xr, yr = box
    w = xr - xl + 1
    h = yr - yl + 1
    # drop too small or out of image boxes
    if min(w, h) < 20 or xl < 0 or yl < 0:
        return False
    return True


def get_p_net_data(data_dir, p_net_size, output_files):
    # face annotations
    annotation_file = os.path.join(data_dir, 'wider_face_train.txt')
    # images
    image_dir = data_dir + '/WIDER_train/images'

    # p net data dir
    save_dir = os.path.join(data_dir, 'p_net')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    pos_file, part_file, neg_file = [open(f, 'w') for f in output_files]

    total_box_num = 0
    with open(annotation_file, 'r') as f:
        annotations = f.readlines()
    image_num = len(annotations)
    for line in annotations:
        annotation = line.strip().split(' ')
        box_num_this_image = len(annotation) // 4
        total_box_num += box_num_this_image
    print('total images: %d, total box num %s.' % (image_num, total_box_num))

    # pos part neg number derived from one box
    pos_and_part_per_box, neg_per_box, neg_per_image = 20, 5, 50
    pos_count, part_count, neg_count = 0, 0, 0
    for line in tqdm(annotations):
        annotation = line.strip().split(' ')
        image_path = annotation[0]

        image = cv2.imread(os.path.join(image_dir, image_path + '.jpg'))
        height, width, channel = image.shape
        # write image path
        for f in [pos_file, part_file, neg_file]:
            f.write(os.path.join(image_dir, image_path + '.jpg') + '\n')

        box = list(map(float, annotation[1:]))
        boxes = np.array(box, dtype=np.float32).reshape(-1, 4)

        neg_amount = 0
        # sample some neg images
        while neg_amount < neg_per_image:
            # random crop size
            size = np.random.randint(p_net_size, min(width, height) / 2)
            # random top-left point
            xl = np.random.randint(0, width - size)
            yl = np.random.randint(0, height - size)
            # crop box
            crop_box = np.array([xl, yl, xl + size, yl + size])
            iou_value = iou(crop_box, boxes)
            # neg if iou < 0.3
            if np.max(iou_value) < 0.3:
                # write cropped box info
                # 0 represents neg
                neg_file.write('%d %d %d %d 0\n' % (crop_box[0], crop_box[1],
                                                    crop_box[2], crop_box[3]))
                neg_amount += 1
                neg_count += 1

        for box in boxes:
            if not is_valid_box(box):
                continue
            xl, yl, xr, yr = box
            w = xr - xl + 1
            h = yr - yl + 1
            for neg_around_box in range(neg_per_box):  # 5 boxes with random offset
                size = np.random.randint(p_net_size, min(width, height) / 2)
                # generate random offset regard to xl and yl
                # -size is ok, but max(-size, -xl) avoid xl_crop < 0
                # ok means the range xl+delta_x:xl+delta_x+size will overlap with box
                delta_x = np.random.randint(max(-size, -xl), w)
                delta_y = np.random.randint(max(-size, -yl), h)
                # top-left corner
                xl_crop = int(max(0, xl + delta_x))
                yl_crop = int(max(0, yl + delta_y))
                # drop too large
                if xl_crop + size > width or yl_crop + size > height:
                    continue
                crop_box = np.array([xl_crop, yl_crop, xl_crop + size, yl_crop + size])
                iou_value = iou(crop_box, boxes)
                if np.max(iou_value) < 0.3:
                    neg_file.write('%d %d %d %d 0\n' % (crop_box[0], crop_box[1],
                                                        crop_box[2], crop_box[3]))
                    neg_count += 1

            # random select pos and part cropped images
            # print('box {}'.format(box))
            for pos_part_around_box in range(pos_and_part_per_box):
                # shrink size
                size = np.random.randint(int(min(w, h)*0.8), int(1.25*max(w, h)))
                delta_x = np.random.randint(-w * 0.2, w * 0.2)
                delta_y = np.random.randint(-h * 0.2, h * 0.2)
                # offset the cropped box's center regard to the center true box along width and height
                xl_crop = int(max(xl + w / 2 + delta_x - size / 2, 0))
                yl_crop = int(max(yl + h / 2 + delta_y - size / 2, 0))
                xr_crop = xl_crop + size - 1
                yr_crop = yl_crop + size - 1
                if xr_crop > width or yr_crop > height:
                    continue
                crop_box = np.array([xl_crop, yl_crop, xr_crop, yr_crop])
                iou_value = iou(crop_box, box.reshape(1, -1))
                # offset, relatively!
                offset_xl = (xl - xl_crop) / float(size)
                offset_yl = (yl - yl_crop) / float(size)
                offset_xr = (xr - xr_crop) / float(size)
                offset_yr = (yr - yr_crop) / float(size)
                # pos image if iou > 0.65
                if iou_value >= 0.65:
                    # 1 represents pos
                    pos_file.write('%d %d %d %d 1 %.2f %.2f %.2f %.2f\n' %
                                   (xl_crop, yl_crop, xr_crop, yr_crop,
                                    offset_xl, offset_yl, offset_xr, offset_yr))
                    pos_count += 1
                elif iou_value >= 0.4:
                    # -1 represents pos
                    part_file.write('%d %d %d %d -1 %.2f %.2f %.2f %.2f\n' %
                                    (xl_crop, yl_crop, xr_crop, yr_crop,
                                     offset_xl, offset_yl, offset_xr, offset_yr))
                    part_count += 1
    print('Processed pos：%s  part: %s neg:%s' % (pos_count, part_count, neg_count))
    for f in [pos_file, part_file, neg_file]:
        f.close()

