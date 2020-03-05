# coding=utf-8

import os
import pickle
from tqdm import tqdm
from preprocess.utils import *
from detector.model import p_net, r_net
from detector.mtcnn_detector import MtCnnDetector
from config import r_net_size, o_net_size, \
    min_face, p_net_stride, face_thresholds


physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def gen_hard_example(size, gray_flag, data_dir, model_paths):
    """
    generate hard example for next net
    :return:
    """
    channel = 1 if gray_flag else 3
    # models

    if size == r_net_size:
        net = 'r_net'
        save_size = r_net_size
    elif size == o_net_size:
        net = 'o_net'
        save_size = o_net_size
    else:
        return
    # images path
    image_dir = os.path.join(data_dir, 'WIDER_train/images/')
    output_dir = os.path.join(data_dir, net)
    detectors = [None, None, None]
    net_p = p_net(channel)
    net_p.load_weights(model_paths[0])
    detectors[0] = net_p
    if size == o_net_size:
        net_r = r_net(channel)
        net_r.load_weights(model_paths[1])
        detectors[1] = net_r
        print("r_net loaded!")
    wider_face_file = os.path.join(data_dir, 'wider_face_train_bbx_gt.txt')
    data = read_annotations(image_dir, wider_face_file)
    mtcnn = MtCnnDetector(detectors, min_face, p_net_stride, face_thresholds)
    save_detects_file = os.path.join(output_dir, net + '_detections.pkl')
    if not os.path.exists(save_detects_file):
        print('loading data to dataset')
        loaded_dataset = load_data_to_dataset(data['images'], channel)
        print('starting to detect')
        detect_result, _ = mtcnn.detect_face(loaded_dataset)  # fixme Not only 3000
        print('detect over')
        with open(save_detects_file, 'wb') as f:
            pickle.dump(detect_result, f, 1)
    print('start to generate hard image')
    save_hard_example(save_size, data, save_detects_file, output_dir)


def save_hard_example(save_size, data, saved_file, output_dir):
    """
    crop original image using previous net outputted boxes for next net
    :param save_size:
    :param data:
    :param saved_file:
    :param output_dir:
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
        [output_dir+"/train_%s_%s.txt" % (net, i) for i in ['neg', 'pos', 'part']]
    neg_file, pos_file, part_file = \
        [open(file, 'w') for file in [neg_label_file, pos_label_file, part_label_file]]
    # read detect results
    with open(saved_file, 'rb') as sf:
        detected_box = pickle.load(sf)
    print('num of detected_box is {}, num of images is {}.'.format(
        len(detected_box), num_img))
    neg_idx, pos_idx, part_idx = 0, 0, 0
    proc_idx = 0
    assert len(img_list) == len(detected_box) == len(gt_boxes_list), "wrong number!"
    for img_idx, detect_box, gt_box in tqdm(zip(img_list, detected_box, gt_boxes_list)):
        gt_box = np.array(gt_box, dtype=np.float32).reshape(-1, 4)
        proc_idx += 1
        if detect_box.shape[0] == 0:
            continue
        img = cv2.imread(img_idx)
        detect_box = convert_to_square(detect_box)
        detect_box[:, :4] = np.round(detect_box[:, :4])
        for f in [pos_file, part_file, neg_file]:
            f.write(img_idx + '\n')
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
                neg_file.write("%d %d %d %d 0\n" % (xl, yl, xr, yr))
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
                    pos_file.write('%d %d %d %d 1 %.2f %.2f %.2f %.2f\n' %
                                   (xl, yl, xr, yr, offset_xl, offset_yl, offset_xr, offset_yr))
                    pos_idx += 1
                elif np.max(iou_value) >= 0.4:
                    part_file.write('%d %d %d %d -1 %.2f %.2f %.2f %.2f\n' %
                                    (xl, yl, xr, yr, offset_xl, offset_yl, offset_xr, offset_yr))
                    part_idx += 1
        if proc_idx >= min(len(detected_box), num_img):
            break
    neg_file.close()
    part_file.close()
    pos_file.close()


# def parse_arguments(argv):
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('--input_size', type=int,
#                         help='The input size for specific net')
#     parser.add_argument('--gray_input', type=bool, default=True)
#     return parser.parse_args(argv)
#
#
# if __name__ == '__main__':
#     main(parse_arguments(sys.argv[1:]))
