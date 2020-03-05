# coding=utf-8

import os
import sys
import tensorflow as tf
import cv2
from tqdm import tqdm
import argparse
import pickle
from config import p_net_size, r_net_size, o_net_size
from preprocess.gen_p_net_data_in_one import get_p_net_data
from preprocess.gen_landmark_augment import gen_landmark
from preprocess.gen_hard_example import gen_hard_example


def main(args):
    """
    generate tfrecord files
    :param args:
    :return:
    """
    data_dir = './data'
    size = args.input_size
    gray_flag = args.gray_input
    if size == p_net_size:
        net = 'p_net'
    elif size == r_net_size:
        net = 'r_net'
    elif size == o_net_size:
        net = 'o_net'
    else:
        return
    # output tfrecord dir
    output_dir = os.path.join(data_dir, net)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # 4 for one net respectively
    if size == p_net_size:
        tf_file_names = [os.path.join(output_dir, 'train_p_net_pos.tfrecord'),
                         os.path.join(output_dir, 'train_p_net_part.tfrecord'),
                         os.path.join(output_dir, 'train_p_net_neg.tfrecord'),
                         os.path.join(output_dir, 'train_p_net_landmark.tfrecord')]
        read_files = [os.path.join(output_dir, 'train_p_net_pos.txt'),
                      os.path.join(output_dir, 'train_p_net_part.txt'),
                      os.path.join(output_dir, 'train_p_net_neg.txt'),
                      os.path.join(output_dir, 'train_p_net_landmark.txt'),
                      os.path.join(output_dir, 'train_p_net_landmark.pkl')]
        for idx, tf_name in enumerate(tf_file_names[:3]):
            if not tf.io.gfile.exists(tf_name):
                if not os.path.exists(read_files[idx]):
                    print('starting to generate %s, %s, %s' % (read_files[0],
                                                               read_files[1],
                                                               read_files[2]))
                    get_p_net_data(data_dir, size, read_files[:3])
                print('starting to write %s...' % tf_file_names[idx])
                write_tfrecord(tf_name, size, read_files[idx], gray_flag)

        # landmark file
        if not tf.io.gfile.exists(tf_file_names[3]):
            if not os.path.exists(read_files[3]) or not os.path.exists(read_files[4]):
                print('starting to generate landmark examples for p_net...')
                gen_landmark(data_dir, size, read_files[3], read_files[4], gray_flag)
            print('starting to write pos/part/neg/landmark mixed tfrecord for p_net...')
            write_tfrecord(tf_file_names[3], size, read_files[3], gray_flag, read_files[4])
            print('p_net mixed tfrecord transform done')

    elif size == r_net_size or size == o_net_size:
        tf_file_names = [os.path.join(output_dir, 'train_%s_%s.tfrecord' % (net, _))
                         for _ in ['pos', 'part', 'neg', 'landmark']]
        read_files = [os.path.join(output_dir, 'train_%s_%s.txt' % (net, _))
                      for _ in ['pos', 'part', 'neg', 'landmark']]
        read_files.append(os.path.join(output_dir, 'train_%s_landmark.pkl' % net))
        model_paths = ['./models/p_net/p_net_30', './models/r_net/r_net_22']
        for idx, tf_name in enumerate(tf_file_names[:3]):
            if not tf.io.gfile.exists(tf_name):
                if not os.path.exists(read_files[idx]):
                    print('starting to generate %s, %s, %s' % (read_files[0],
                                                               read_files[1],
                                                               read_files[2]))
                    gen_hard_example(size, gray_flag, data_dir, model_paths)
                print('starting to write %s...' % tf_file_names[idx])
                write_tfrecord(tf_name, size, read_files[idx], gray_flag)
        # landmark file
        if not tf.io.gfile.exists(tf_file_names[3]):
            if not os.path.exists(read_files[3]) or not os.path.exists(read_files[4]):
                print('starting to generate landmark examples for p_net...')
                gen_landmark(data_dir, size, read_files[3], read_files[4], gray_flag)
            print('starting to write pos/part/neg/landmark mixed tfrecord for p_net...')
            write_tfrecord(tf_file_names[3], size, read_files[3], gray_flag, read_files[4])
            print('p_net mixed tfrecord transform done')
    else:
        return


def write_tfrecord(tf_file, size, txt_file, gray, pkl_file=None):
    print('start reading data')
    if pkl_file is not None:
        dataset = get_landmark_dataset(txt_file, pkl_file)
    else:
        dataset = get_dataset(txt_file, size, gray)
    # starting write tfrecord
    with tf.io.TFRecordWriter(tf_file) as tf_writer:
        for image_info_dict in tqdm(dataset):
            _add_to_tfrecord(image_info_dict, tf_writer)


def get_landmark_dataset(data_file, pkl_file):
    pkl = open(pkl_file, 'rb')
    face_images = pickle.load(pkl)
    print('there is %s faces with landmark' % len(face_images))
    landmark_file = open(data_file, 'r')
    dataset = []
    for line in tqdm(landmark_file.readlines()):
        data_example = dict()
        landmark_info = line.strip().split(' ')
        image = face_images[int(landmark_info[0])]
        data_example['image'] = image.tostring()
        data_example['label'] = int(landmark_info[1])
        bbox = dict()
        bbox['xmin'] = 0
        bbox['ymin'] = 0
        bbox['xmax'] = 0
        bbox['ymax'] = 0
        bbox['xlefteye'] = float(landmark_info[2])
        bbox['ylefteye'] = float(landmark_info[3])
        bbox['xrighteye'] = float(landmark_info[4])
        bbox['yrighteye'] = float(landmark_info[5])
        bbox['xnose'] = float(landmark_info[6])
        bbox['ynose'] = float(landmark_info[7])
        bbox['xleftmouth'] = float(landmark_info[8])
        bbox['yleftmouth'] = float(landmark_info[9])
        bbox['xrightmouth'] = float(landmark_info[10])
        bbox['yrightmouth'] = float(landmark_info[11])
        data_example['bbox'] = bbox
        dataset.append(data_example)
    landmark_file.close()
    pkl.close()
    return dataset


def get_dataset(data_file, size, gray):
    """
    get data from txt file
    :param data_file: data_file
    :param size: size for resize
    :param gray:
    :return:
    """
    image_list = open(data_file, 'r')
    dataset = []
    image = None
    for line in tqdm(image_list.readlines()):
        line = line.strip().split(' ')
        if (len(line)) == 1:
            image = cv2.imread(line[0])
        else:
            data_example = dict()
            xl, yl, xr, yr = [int(line[i]) for i in range(4)]
            cropped_img = cv2.resize(image[yl:yr+1, xl:xr+1, :], (size, size),
                                     cv2.INTER_LINEAR)
            if gray:
                cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
            data_example['image'] = cropped_img.tostring()
            data_example['label'] = int(line[4])
            bbox = dict()
            bbox['xmin'] = 0
            bbox['ymin'] = 0
            bbox['xmax'] = 0
            bbox['ymax'] = 0
            bbox['xlefteye'] = 0
            bbox['ylefteye'] = 0
            bbox['xrighteye'] = 0
            bbox['yrighteye'] = 0
            bbox['xnose'] = 0
            bbox['ynose'] = 0
            bbox['xleftmouth'] = 0
            bbox['yleftmouth'] = 0
            bbox['xrightmouth'] = 0
            bbox['yrightmouth'] = 0
            if data_example['label'] != 0:
                bbox['xmin'] = float(line[5])
                bbox['ymin'] = float(line[6])
                bbox['xmax'] = float(line[7])
                bbox['ymax'] = float(line[8])
            data_example['bbox'] = bbox
            dataset.append(data_example)
    image_list.close()
    return dataset


def _add_to_tfrecord(image_dict, tfrecord_writer):
    """
    write to tfrecord
    :param image_dict: contains image ground truth info
    :param tfrecord_writer:
    :return:
    """
    '''转换成tfrecord文件
    参数：
      filename：图片文件名
      image_example:数据
      tfrecord_writer:写入文件
    '''
    # image_data, height, width = _process_image_without_coder(image_dict['filename'])
    example = _convert_to_example_simple(image_dict)
    tfrecord_writer.write(example.SerializeToString())


# def _process_image_without_coder(filename):
#     """
#     read image file
#     :param filename:
#     :return:
#     """
#     image = cv2.imread(filename)  # fixme!!! change to read image and box, then crop, resize
#     image_data = image.tostring()
#     assert len(image.shape) == 3
#     height = image.shape[0]
#     width = image.shape[1]
#     assert image.shape[2] == 3
#     return image_data, height, width


def _convert_to_example_simple(image_dict):
    """
    convert image to tfrecord format
    :param image_dict:
    :return:
    """
    image_data = image_dict['image']
    class_label = image_dict['label']
    bbox = image_dict['bbox']
    roi = [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']]
    landmark = [bbox['xlefteye'], bbox['ylefteye'],
                bbox['xrighteye'], bbox['yrighteye'],
                bbox['xnose'], bbox['ynose'],
                bbox['xleftmouth'], bbox['yleftmouth'],
                bbox['xrightmouth'], bbox['yrightmouth']]

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(image_data),
        'image/label': _int64_feature(class_label),
        'image/roi': _float_feature(roi),
        'image/landmark': _float_feature(landmark)
    }))
    return example


# convert data format
def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_size', type=int,
                        help='The input size for specific net')
    parser.add_argument('--gray_input', type=bool, nargs='?',
                        default=True)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
