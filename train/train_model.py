# coding=utf-8

import os
import sys
sys.path.append('../')
print(sys.path)
import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf
import train_config as tc
import random
import cv2
from model import *
from config import p_net_size, r_net_size, o_net_size

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def train(args):
    size = args.input_size
    nets = {p_net_size: 'p_net',
            r_net_size: 'r_net',
            o_net_size: 'o_net'}
    loss_weights = {'p_net': [1.0, 0.5, 0.5],
                    'r_net': [1.0, 0.5, 0.5],
                    'o_net': [1.0, 0.5, 1.0]}
    data_dir = '../data'

    net = nets[size]
    loss_wight_face, loss_wight_bbox, loss_weight_landmark = loss_weights[net]
    if size == p_net_size:
        # read number of data
        label_file = os.path.join(data_dir, net + '/train_p_net_landmark.txt')
        with open(label_file, 'r') as f:
            num = len(f.readlines())
        dataset_file = os.path.join(data_dir, net + '/tfrecord/train_p_net_landmark.tfrecord')
        end_epoch = tc.end_epoch[0]
        dataset = read_single_tfrecord(dataset_file, tc.batch_size, size, num)
    else:
        label_files = [os.path.join(data_dir, net + "/%s_%s.txt" % (net, i))
                       for i in ['pos', 'part', 'neg']]
        label_files.append(os.path.join(data_dir, net + "/landmark_%s_aug.txt" % net))
        #
        nums = []
        for lf in label_files:
            with open(lf, 'r') as f:
                nums.append(len(f.readlines()))
        num = sum(nums)
        dataset_files = [os.path.join(data_dir, net+'/tfrecord/%s_landmark.tfrecord' % s)
                         for s in ['pos', 'part', 'neg', 'landmark']]
        pos_ratio, part_ratio, neg_ratio, landmark_ratio = 1.0/6, 1.0/6, 3.0/6, 1.0/6
        batch_sizes = [np.ceil(tc.batch_size * ratio)
                       for ratio in [pos_ratio, part_ratio, neg_ratio, landmark_ratio]]
        end_epoch = tc.end_epoch[1] if size == r_net_size else tc.end_epoch[2]
        dataset = read_multi_tfrecords(dataset_files, batch_sizes, size, nums)
    model = get_model(size)
    lr_factor = 0.1
    global_step = tf.Variable(0, trainable=False)
    boundaries = [int(epoch * num / tc.batch_size) for epoch in tc.LR_EPOCH]
    lr_values = [tc.lr * (lr_factor ** x) for x in range(len(tc.LR_EPOCH) + 1)]
    lr_op = tf.compat.v1.train.piecewise_constant(global_step, boundaries, lr_values)
    optimizer = tf.compat.v1.train.MomentumOptimizer(lr_op, 0.9)
    step = 0
    epoch = 0
    max_step = int(num / tc.batch_size + 1) * end_epoch
    for image, label, roi, landmark in dataset:
        # print('image batch shape = {}'.format(image.get_shape()))
        step += 1
        image_flipped, landmark_flipped = random_flip_images(image, label, landmark)
        # TODO uncomment above sentence
        # print('image shape = {}'.format(image[0].get_shape()))
        input_image = image_color_distort(image_flipped)

        loss_value, accuracy, cls_loss, bbox_loss, lm_loss, reg_loss, grads = \
            grad(model, input_image, label, roi, landmark_flipped,
                 (loss_wight_face, loss_wight_bbox, loss_weight_landmark))
        optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step=global_step)
        if (step + 1) % tc.display == 0:
            print('epoch:%d/%d' % (epoch + 1, end_epoch))
            print(
                "Step: %d/%d, accuracy: %3f, cls loss: %4f, bbox loss: %4f, "
                "Landmark loss :%4f, L2 loss: %4f, Total Loss: %4f " % (
                    step + 1, max_step, accuracy, cls_loss, bbox_loss, lm_loss, reg_loss, loss_value))
        if step * tc.batch_size > num * (epoch+1):
            epoch += 1
            model.save_weights("../models/%s/" % net)
        if step > max_step:
            break


def get_model(size):
    if size == p_net_size:
        model = p_net()
    elif size == r_net_size:
        model = r_net()
    elif size == o_net_size:
        model = o_net()
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
    reg_los = tf.add_n(model.losses)
    loss_all = cls_loss * weights[0] + bbox_loss * weights[1] + landmark_loss * weights[2] + reg_los
    accuracy = cal_accuracy(predict_face, label)
    # print('cls loss={}, bbox loss={}, landmark loss={}, reg loss={}'.format(
    #     cls_loss, bbox_loss, landmark_loss, reg_los))
    return loss_all, accuracy, cls_loss, bbox_loss, landmark_loss, reg_los


def grad(model, inputs, label, bbox_gt, landmark_gt, weights):
    with tf.GradientTape() as tape:
        loss_value, acc, cls_loss, bbox_loss, lm_loss, reg_loss = \
            total_loss(model, inputs, label, bbox_gt, landmark_gt, weights)
    return loss_value, acc,cls_loss, bbox_loss, lm_loss, reg_loss,\
           tape.gradient(loss_value, model.trainable_variables)


def read_single_tfrecord(tfrecord_file, batch_size, image_size, num):
    """
    read data from tfrecord file
    :param tfrecord_file:
    :param batch_size:
    :param image_size:
    :param num:
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
        image = tf.reshape(image, [image_size, image_size, 3])
        image = (tf.cast(image, tf.float32) - 127.5) / 128

        label = tf.cast(parsed_example['image/label'], tf.float32)
        roi = tf.cast(parsed_example['image/roi'], tf.float32)
        landmark = tf.cast(parsed_example['image/landmark'], tf.float32)
        return image, label, roi, landmark

    parsed_dataset = dataset.map(_parse_function).batch(batch_size)
    return parsed_dataset


def read_multi_tfrecords(tf_files, batch_sizes, size, nums):
    """
    read multi tfrecord files together
    :param tf_files:
    :param batch_sizes: batch size in each category
    :param size:
    :param nums:
    :return:
    """
    dataset_pos = read_single_tfrecord(tf_files[0], batch_sizes[0], size, nums[0])
    dataset_part = read_single_tfrecord(tf_files[1], batch_sizes[1], size, nums[1])
    dataset_neg = read_single_tfrecord(tf_files[2], batch_sizes[2], size, nums[2])
    dataset_landmark = read_single_tfrecord(tf_files[3], batch_sizes[3], size, nums[3])
    while True:
        pos_item = next(iter(dataset_pos))
        part_item = next(iter(dataset_part))
        neg_item = next(iter(dataset_neg))
        landmark_item = next(iter(dataset_landmark))
        yield (tf.concat([pos_item[0], part_item[0], neg_item[0], landmark_item[0]], 0),
               tf.concat([pos_item[1], part_item[1], neg_item[1], landmark_item[1]], 0),
               tf.concat([pos_item[2], part_item[2], neg_item[2], landmark_item[2]], 0),
               tf.concat([pos_item[3], part_item[3], neg_item[3], landmark_item[3]], 0))


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
        flip_pos_indexes = np.where(label_batch == 1)[0]

        flip_indexes = np.concatenate((flip_landmark_indexes, flip_pos_indexes))

        for i in flip_indexes:
            cv2.flip(image_batch[i], 1, image_batch[i])

        for i in flip_landmark_indexes:
            landmark_ = landmark_batch[i].reshape((-1, 2))
            landmark_ = np.asarray([(1 - x, y) for (x, y) in landmark_])
            landmark_[[0, 1]] = landmark_[[1, 0]]
            landmark_[[3, 4]] = landmark_[[4, 3]]
            landmark_batch[i] = landmark_.ravel()

    return image_batch, landmark_batch


def image_color_distort(inputs):
    inputs = tf.image.random_contrast(inputs, lower=0.5, upper=1.5)
    inputs = tf.image.random_brightness(inputs, max_delta=0.2)
    inputs = tf.image.random_hue(inputs, max_delta=0.2)
    inputs = tf.image.random_saturation(inputs, lower=0.5, upper=1.5)

    return inputs


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_size', type=int,
                        help='The input size for specific net')

    return parser.parse_args(argv)


if __name__ == '__main__':
    train(parse_arguments(sys.argv[1:]))
