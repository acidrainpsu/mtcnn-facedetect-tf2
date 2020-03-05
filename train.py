# coding=utf-8

import os
import sys
import argparse
import numpy as np
import random
import cv2
from detector.model import *
from config import *

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def train(args):
    size = args.input_size
    gray_flag = args.gray_input
    nets = {p_net_size: 'p_net',
            r_net_size: 'r_net',
            o_net_size: 'o_net'}
    loss_weights = {'p_net': [1.0, 0.5, 0.5],
                    'r_net': [1.0, 0.5, 0.5],
                    'o_net': [1.0, 0.5, 1.0]}
    net = nets[size]
    data_dir = './data'
    model_save_dir = './models/'
    model_save_path = model_save_dir + net + '/' + net+'_{}'
    net_data_dir = os.path.join(data_dir, net)
    loss_wight_face, loss_wight_bbox, loss_weight_landmark = loss_weights[net]

    dataset_files = [os.path.join(net_data_dir, 'train_%s_%s.tfrecord' % (net, cat))
                     for cat in ['pos', 'part', 'neg', 'landmark']]
    sample_nums = calc_sample_nums(data_dir, net)
    pos_ratio, part_ratio, neg_ratio, landmark_ratio = 1.0/6, 1.0/6, 3.0/6, 1.0/6
    batch_sizes = [np.ceil(batch_size * ratio)
                   for ratio in [pos_ratio, part_ratio, neg_ratio, landmark_ratio]]
    epoch = end_epochs[0] if size == p_net_size else \
        end_epochs[1] if size == r_net_size else end_epochs[2]
    dataset = read_multi_tfrecords(dataset_files, batch_sizes, size, sample_nums, gray_flag)
    sample_num_sum = sum(sample_nums)
    print('total sample num = %d' % sample_num_sum)
    model = get_model(size, gray_flag)
    lr_factor = 0.1
    global_step = tf.Variable(0, trainable=False)
    boundaries = [int(_epoch * sample_num_sum / batch_size) for _epoch in LR_EPOCH]
    lr_values = [lr * (lr_factor ** x) for x in range(len(LR_EPOCH) + 1)]
    lr_op = tf.compat.v1.train.piecewise_constant(global_step, boundaries, lr_values)
    optimizer = tf.compat.v1.train.MomentumOptimizer(lr_op, 0.9)
    step = 0
    epoch_idx = 0
    max_step = int(sample_num_sum / batch_size + 1) * epoch
    for batch in dataset:
        # print('image batch shape = {}'.format(image.get_shape()))
        image, label, roi, landmark = reassemble_data(batch)
        # print('image batch shape = {}, pixel = {}'.format(image.get_shape(), image.numpy()[0, 0, :]))
        step += 1
        image_flipped, landmark_flipped = random_flip_images(image, label, landmark)  # TODO: add in random flip
        # print('image shape = {}'.format(image[0].get_shape()))
        input_image = image_color_distort(image_flipped, gray_flag)
        # if len(input_image.get_shape()) < 4:
        #     input_image = tf.expand_dims(input_image, -1)
        # print('image batch shape = {}'.format(input_image.get_shape()))

        loss_value, accuracy, cls_loss, bbox_loss, lm_loss, grads = \
            grad(model, input_image, label, roi, landmark_flipped,
                 (loss_wight_face, loss_wight_bbox, loss_weight_landmark))
        optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step=global_step)
        if (step + 1) % display == 0:
            print('epoch:%d/%d' % (epoch_idx + 1, epoch))
            print(
                "Step: %d/%d, accuracy: %3f, cls loss: %4f, bbox loss: %4f, "
                "Landmark loss :%4f, Total Loss: %4f " % (
                    step + 1, max_step, accuracy, cls_loss, bbox_loss, lm_loss, loss_value))
        if step * batch_size > sample_num_sum * (epoch_idx+1):
            epoch_idx += 1
            if epoch_idx > epoch-3:
                model.save_weights(model_save_path.format(epoch_idx))
        if step > max_step:
            break


def get_model(size, gray_flag):
    if gray_flag:
        channel = 1
    else:
        channel = 3
    if size == p_net_size:
        model = p_net(channel)
    elif size == r_net_size:
        model = r_net(channel)
    elif size == o_net_size:
        model = o_net(channel)
    else:
        return None
    return model


def total_loss(model, inputs, label, bbox_gt, landmark_gt, weights):
    # print('total loss : inputs batch size = {}'.format(inputs.get_shape()))
    predict_face, predict_bbox, predict_landmark = model(inputs)
    # print('total loss : face size = {}, bbox size={}, landmark size={}'.format(
    #     predict_face.get_shape(), predict_bbox.get_shape(), predict_landmark.get_shape()))
    cls_loss = cls_ohem(predict_face, label)
    bbox_loss = bbox_ohem(predict_bbox, bbox_gt, label)
    landmark_loss = landmark_ohem(predict_landmark, landmark_gt, label)
    # reg_los = tf.add_n(model.losses) Fixme: add reg loss back
    loss_all = cls_loss * weights[0] + bbox_loss * weights[1] + landmark_loss * weights[2]  # + reg_los
    accuracy = cal_accuracy(predict_face, label)
    # print('cls loss={}, bbox loss={}, landmark loss={}, reg loss={}'.format(
    #     cls_loss, bbox_loss, landmark_loss, reg_los))
    return loss_all, accuracy, cls_loss, bbox_loss, landmark_loss  # , reg_los


def grad(model, inputs, label, bbox_gt, landmark_gt, weights):
    with tf.GradientTape() as tape:  # Fixme : reg loss
        loss_value, acc, cls_loss, bbox_loss, lm_loss = \
            total_loss(model, inputs, label, bbox_gt, landmark_gt, weights)
    return loss_value, acc, cls_loss, bbox_loss, lm_loss, \
           tape.gradient(loss_value, model.trainable_variables)


def read_single_tfrecord(tfrecord_file, sub_batch_size, image_size, num, channel):
    """
    read data from tfrecord file
    :param tfrecord_file:
    :param sub_batch_size:
    :param image_size:
    :param num: sample num
    :return:
    """
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    print('example num= %d' % num)
    dataset = dataset.shuffle(num, reshuffle_each_iteration=True)
    dataset = dataset.repeat()
    example_features = {'image/encoded': tf.io.FixedLenFeature([], tf.string),
                        'image/label': tf.io.FixedLenFeature([], tf.int64),
                        'image/roi': tf.io.FixedLenFeature([4], tf.float32),
                        'image/landmark': tf.io.FixedLenFeature([10], tf.float32)}

    def _parse_function(example_proto):
        parsed_example = tf.io.parse_single_example(example_proto, example_features)
        image = tf.io.decode_raw(parsed_example['image/encoded'], tf.uint8)
        image = tf.reshape(image, [image_size, image_size, channel])
        image = (tf.cast(image, tf.float32) - 127.5) / 128

        label = tf.cast(parsed_example['image/label'], tf.float32)
        roi = tf.cast(parsed_example['image/roi'], tf.float32)
        landmark = tf.cast(parsed_example['image/landmark'], tf.float32)
        return image, label, roi, landmark

    parsed_dataset = dataset.map(_parse_function).batch(sub_batch_size)
    return parsed_dataset


def read_multi_tfrecords(tf_files, batch_sizes, size, sample_nums, gray_flag):
    """
    read multi tfrecord files together
    :param tf_files:
    :param batch_sizes: batch size in each category
    :param size:
    :param sample_nums: num of each category
    :return:
    """
    channel = 1 if gray_flag else 3
    dataset_pos = read_single_tfrecord(tf_files[0], batch_sizes[0], size, sample_nums[0], channel)
    dataset_part = read_single_tfrecord(tf_files[1], batch_sizes[1], size, sample_nums[1], channel)
    dataset_neg = read_single_tfrecord(tf_files[2], batch_sizes[2], size, sample_nums[2], channel)
    dataset_landmark = read_single_tfrecord(tf_files[3], batch_sizes[3], size, sample_nums[3], channel)
    return tf.data.Dataset.zip((dataset_pos, dataset_part, dataset_neg, dataset_landmark))


def reassemble_data(zipped_element):
    pos_item, part_item, neg_item, landmark_item = zipped_element
    image_data = tf.concat([pos_item[0], part_item[0], neg_item[0], landmark_item[0]], 0)
    label_data = tf.concat([pos_item[1], part_item[1], neg_item[1], landmark_item[1]], 0)
    roi_data = tf.concat([pos_item[2], part_item[2], neg_item[2], landmark_item[2]], 0)
    landmark_data = tf.concat([pos_item[3], part_item[3], neg_item[3], landmark_item[3]], 0)
    return image_data, label_data, roi_data, landmark_data


def random_flip_images(image_batch, label_batch, landmark_batch):
    """
    flip batch images randomly
    :param image_batch:
    :param label_batch:
    :param landmark_batch:
    :return:
    """
    image_batch, label_batch, landmark_batch = \
        image_batch.numpy(), label_batch.numpy(), landmark_batch.numpy()
    if random.choice([0, 1]) > 0:
        num_images = image_batch.shape[0]
        flip_landmark_indexes = np.where(label_batch == -2)[0]
        # flip_pos_indexes = np.where(label_batch == 1)[0]  # TODO: mark this
        # flip_indexes = np.concatenate((flip_landmark_indexes, flip_pos_indexes))

        for i in flip_landmark_indexes:  # TODO: mark this
            cv2.flip(image_batch[i], 1, image_batch[i])

        for i in flip_landmark_indexes:
            landmark_ = landmark_batch[i].reshape((-1, 2))
            landmark_ = np.asarray([(1 - x, y) for (x, y) in landmark_])
            landmark_[[0, 1]] = landmark_[[1, 0]]
            landmark_[[3, 4]] = landmark_[[4, 3]]
            landmark_batch[i] = landmark_.ravel()

    return image_batch, landmark_batch


def image_color_distort(inputs, gray_flag):
    inputs = tf.image.random_contrast(inputs, lower=0.5, upper=1.5)
    inputs = tf.image.random_brightness(inputs, max_delta=0.2)
    if not gray_flag:
        inputs = tf.image.random_hue(inputs, max_delta=0.2)
        inputs = tf.image.random_saturation(inputs, lower=0.5, upper=1.5)
    return inputs


def calc_sample_nums(data_dir, net):
    image_file = os.path.join(data_dir, "wider_face_train.txt")
    with open(image_file, 'r') as f:
        image_num = len(f.readlines())
    example_files = [os.path.join(data_dir, net + '/train_%s_%s.txt' % (net, cat))
                     for cat in ['pos', 'part', 'neg', 'landmark']]
    opened_file = [open(file, 'r') for file in example_files]
    pos_num, part_num, neg_num, landmark_num = [len(of.readlines()) for of in opened_file]
    pos_num -= image_num
    part_num -= image_num
    neg_num -= image_num
    return pos_num, part_num, neg_num, landmark_num


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_size', type=int,
                        help='The input size for specific net')
    parser.add_argument('--gray_input', type=bool, default=True)

    return parser.parse_args(argv)


if __name__ == '__main__':
    train(parse_arguments(sys.argv[1:]))
