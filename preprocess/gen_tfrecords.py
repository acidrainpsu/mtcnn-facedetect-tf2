# coding=utf-8

import os
import sys
sys.path.append('../')
import tensorflow as tf
import cv2
from tqdm import tqdm
import argparse
from config import p_net_size, r_net_size, o_net_size


def main(args):
    """
    generate tfrecords files
    :param args:
    :return:
    """
    size = args.input_size
    if size == p_net_size:
        net = 'p_net'
    elif size == r_net_size:
        net = 'r_net'
    elif size == o_net_size:
        net = 'o_net'
    else:
        net = None
    data_dir = '../data'
    # output tfrecord dir
    output_dir = os.path.join(data_dir, net+'/tfrecord')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # one tfrecord for p_net, 4 for r_net and o_net
    if size == p_net_size:
        tf_file_names = [os.path.join(output_dir,'train_%s_landmark.tfrecord' % net)]
        items = [net+'/train_%s_landmark.txt' % net]
    elif size == r_net_size or size == o_net_size:
        tf_file_names = [os.path.join(output_dir, '%s_landmark.tfrecord' % _)
                         for _ in ['pos', 'part', 'neg', 'landmark']]
        items = ['%s/%s_%s.txt' % (net, net, _) for _ in ['pos', 'part', 'neg']]
        items.append('%s/landmark_%s_aug.txt' % (net, net))
    else:
        tf_file_names = None
        items = None

    if tf.io.gfile.exists(tf_file_names[0]):
        print('tfrecord has been there!!')
        return

    # get data
    for tf_file, item in zip(tf_file_names, items):
        print('start reading data')
        dataset = get_dataset(data_dir, item)
        # starting write tfrecord
        with tf.io.TFRecordWriter(tf_file) as tf_writer:
            for image_example in tqdm(dataset):
                file_name = image_example['filename']
                try:
                    _add_to_tfrecord(file_name, image_example, tf_writer)
                except:
                    print(file_name)
    print('tfrecord transform done')


def get_dataset(data_dir, item):
    """
    get data from txt file
    :param data_dir: data dir
    :param item: txt source
    :return:
    """
    data_file = os.path.join(data_dir, item)
    image_list = open(data_file, 'r')
    dataset = []
    for line in tqdm(image_list.readlines()):
        info = line.strip().split(' ')
        data_example = dict()
        bbox = dict()
        data_example['filename'] = info[0]
        data_example['label'] = int(info[1])
        # box of neg is 0,part,pos contains face box only, landmark contains 5 key points
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
        if len(info) == 6:
            bbox['xmin'] = float(info[2])
            bbox['ymin'] = float(info[3])
            bbox['xmax'] = float(info[4])
            bbox['ymax'] = float(info[5])
        if len(info) == 12:
            bbox['xlefteye'] = float(info[2])
            bbox['ylefteye'] = float(info[3])
            bbox['xrighteye'] = float(info[4])
            bbox['yrighteye'] = float(info[5])
            bbox['xnose'] = float(info[6])
            bbox['ynose'] = float(info[7])
            bbox['xleftmouth'] = float(info[8])
            bbox['yleftmouth'] = float(info[9])
            bbox['xrightmouth'] = float(info[10])
            bbox['yrightmouth'] = float(info[11])
        data_example['bbox'] = bbox
        dataset.append(data_example)
    return dataset


def _add_to_tfrecord(filename, image_dict, tfrecord_writer):
    """
    write to tfrecord
    :param filename: image file path
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
    image_data, height, width = _process_image_without_coder(filename)
    example = _convert_to_example_simple(image_dict, image_data)
    tfrecord_writer.write(example.SerializeToString())


def _process_image_without_coder(filename):
    """
    read image file
    :param filename:
    :return:
    """
    image = cv2.imread(filename)
    image_data = image.tostring()
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3
    return image_data, height, width


def _convert_to_example_simple(image_dict, image_data):
    """
    convert image to tfrecord format
    :param image_dict:
    :param image_data:
    :return:
    """
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

    parser.add_argument('input_size', type=int,
                        help='The input size for specific net')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
