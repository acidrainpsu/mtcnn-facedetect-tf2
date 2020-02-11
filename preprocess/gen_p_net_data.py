# coding=utf-8

import os
import cv2
import numpy as np
from tqdm import tqdm
from utils import iou
from config import p_net_size


# face annotations
annotation_file = '../data/wider_face_train.txt'
# images
image_dir = '../data/WIDER_train/images'
# output dir of pos, part, neg cropped images
pos_save_dir = '../data/p_net/pos'
part_save_dir = '../data/p_net/part'
neg_save_dir = '../data/p_net/neg'
# p net data dir
save_dir = '../data/p_net'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(pos_save_dir):
    os.mkdir(pos_save_dir)
if not os.path.exists(part_save_dir):
    os.mkdir(part_save_dir)
if not os.path.exists(neg_save_dir):
    os.mkdir(neg_save_dir)

f1 = open(os.path.join(save_dir, 'p_net_pos.txt'), 'w')
f2 = open(os.path.join(save_dir, 'p_net_neg.txt'), 'w')
f3 = open(os.path.join(save_dir, 'p_net_part.txt'), 'w')

with open(annotation_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)
print('total images: %d' % num)
# count num of pos, neg, part examples
pos_count = 0
neg_count = 0
part_count = 0
image_count = 0

for annotation in tqdm(annotations):
    annotation = annotation.strip().split(' ')
    image_path = annotation[0]
    box = list(map(float, annotation[1:]))

    boxes = np.array(box, dtype=np.float32).reshape(-1, 4)
    image = cv2.imread(os.path.join(image_dir, image_path+'.jpg'))
    image_count += 1
    height, width, channel = image.shape

    neg_amount = 0
    # sample some neg images
    while neg_amount < 50:
        # random crop size
        size = np.random.randint(p_net_size, min(width, height)/2)
        # random top-left point
        nx = np.random.randint(0, width - size)
        ny = np.random.randint(0, height - size)
        # crop box
        crop_box = np.array([nx, ny, nx+size, ny+size])
        iou_value = iou(crop_box, boxes)

        # neg if iou < 0.3
        if np.max(iou_value) < 0.3:
            # crop image and resize to p_net size
            cropped_image = image[ny:ny + size, nx:nx + size, :]
            resized_image = cv2.resize(cropped_image, (p_net_size, p_net_size), interpolation=cv2.INTER_LINEAR)
            save_file = os.path.join(neg_save_dir, '%s.jpg' % neg_count)
            f2.write(str(save_file) + ' 0\n')  # 0 represents neg
            cv2.imwrite(save_file, resized_image)
            neg_count += 1
            neg_amount += 1

    for box in boxes:
        xl, yl, xr, yr = box
        w = xr-xl+1
        h = yr-yl+1
        # drop too small or out of image boxes
        if max(w, h) < 20 or min(w, h) < 5 or xl < 0 or yl < 0:
            continue
        for i in range(5):  # 5 boxes with random offset
            size = np.random.randint(p_net_size, min(width, height)/2)
            # generate random offset regard to xl and yl
            delta_x = np.random.randint(max(-size, -xl), w)  # -size is ok, but max(-size, -xl) avoid xl_crop < 0
            delta_y = np.random.randint(max(-size, -yl), h)
            # top-left corner
            xl_crop = int(max(0, xl+delta_x))
            yl_crop = int(max(0, yl+delta_y))
            # drop too large
            if xl_crop+size > width or yl_crop+size > height:
                continue
            crop_box = np.array([xl_crop, yl_crop, xl_crop+size, yl_crop+size])
            iou_value = iou(crop_box, boxes)
            if np.max(iou_value) < 0.3:
                # crop image and resize to p_net size
                cropped_image = image[yl_crop:yl_crop + size, xl_crop:xl_crop + size, :]
                resized_image = cv2.resize(cropped_image, (p_net_size, p_net_size), interpolation=cv2.INTER_LINEAR)
                save_file = os.path.join(neg_save_dir, '%s.jpg' % neg_count)
                f2.write(str(save_file) + ' 0\n')  # 0 represents neg
                cv2.imwrite(save_file, resized_image)
                neg_count += 1
        # random select pos and part cropped images
        for i in range(20):
            # shrink size
            size = np.random.randint(int(min(w, h)*0.8), np.ceil(1.25*max(w, h)))
            if size < 5:
                continue
            delta_x = np.random.randint(-w*0.2, w*0.2)
            delta_y = np.random.randint(-h*0.2, h*0.2)
            # offset the cropped box's center regard to the center true box along width and height
            xl_crop = int(max(xl+w/2+delta_x-size/2, 0))
            yl_crop = int(max(yl+h/2+delta_y-size/2, 0))
            xr_crop = xl_crop+size
            yr_crop = yl_crop+size
            if xr_crop > width or yr_crop > height:
                continue
            crop_box = np.array([xl_crop, yl_crop, xr_crop, yr_crop])
            iou_value = iou(crop_box, box.reshape(1, -1))
            if iou_value >= 0.4:
                # at least part box, do some job
                offset_xl = (xl - xl_crop)/float(size)
                offset_yl = (yl - yl_crop)/float(size)
                offset_xr = (xr - xr_crop)/float(size)
                offset_yr = (yr - yr_crop)/float(size)

                cropped_image = image[yl_crop:yr_crop, xl_crop:xr_crop, :]
                resized_image = cv2.resize(cropped_image, (p_net_size, p_net_size), interpolation=cv2.INTER_LINEAR)
                # pos image if iou > 0.65
                if iou_value >= 0.65:
                    save_file = os.path.join(pos_save_dir, '%s.jpg' % pos_count)
                    f1.write(str(save_file) + ' 1 %.2f %.2f %.2f %.2f\n' %
                             (offset_xl, offset_yl, offset_xr, offset_yr))  # 1 represents pos
                    cv2.imwrite(save_file, resized_image)
                    pos_count += 1
                else:
                    save_file = os.path.join(part_save_dir, '%s.jpg' % part_count)
                    f3.write(str(save_file) + ' -1 %.2f %.2f %.2f %.2f\n' %
                             (offset_xl, offset_yl, offset_xr, offset_yr))  # -1 represents part
                    cv2.imwrite(save_file, resized_image)
                    part_count += 1

print('%s images processed，pos：%s  part: %s neg:%s' % (image_count, pos_count, part_count, neg_count))
f1.close()
f2.close()
f3.close()
