# coding = utf-8

import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.layers as kl
from config import p_net_size, r_net_size, o_net_size

num_keep_ratio = 0.7


def p_net():
    """
    p_net model
    :return: p_net Model
    """
    inputs = ks.Input((None, None, 3))
    x = kl.Conv2D(10, 3, activation=prelu, kernel_regularizer=ks.regularizers.l2(5e-4))(inputs)
    x = kl.MaxPool2D((2, 2), 2, 'same')(x)
    x = kl.Conv2D(16, 3, activation=prelu, kernel_regularizer=ks.regularizers.l2(5e-4))(x)
    x = kl.Conv2D(32, 3, activation=prelu, kernel_regularizer=ks.regularizers.l2(5e-4))(x)
    # multi-output
    predict_face = kl.Conv2D(2, 1, activation=ks.activations.softmax,
                             kernel_regularizer=ks.regularizers.l2(5e-4))(x)
    predict_bbox = kl.Conv2D(4, 1, kernel_regularizer=ks.regularizers.l2(5e-4))(x)
    predict_landmark = kl.Conv2D(10, 1, kernel_regularizer=ks.regularizers.l2(5e-4))(x)
    model = ks.Model(inputs=inputs,
                     outputs=[predict_face, predict_bbox, predict_landmark])
    return model


def r_net():
    """
    r_net model
    :return: r_net model
    """
    inputs = ks.Input((r_net_size, r_net_size, 3))
    x = kl.Conv2D(28, 3, activation=prelu, kernel_regularizer=ks.regularizers.l2(5e-4))(inputs)
    x = kl.MaxPool2D((3, 3), 2, 'same')(x)
    x = kl.Conv2D(48, 3, activation=prelu, kernel_regularizer=ks.regularizers.l2(5e-4))(x)
    x = kl.MaxPool2D((3, 3), 2, 'valid')(x)
    x = kl.Conv2D(64, 2, activation=prelu, kernel_regularizer=ks.regularizers.l2(5e-4))(x)
    flatten = kl.Flatten()(x)
    fc = kl.Dense(128, activation=prelu, kernel_regularizer=ks.regularizers.l2(5e-4))(flatten)

    predict_face = kl.Dense(2, activation=tf.nn.softmax, kernel_regularizer=ks.regularizers.l2(5e-4))(fc)
    predict_bbox = kl.Dense(4, kernel_regularizer=ks.regularizers.l2(5e-4))(fc)
    predict_landmark = kl.Dense(10, kernel_regularizer=ks.regularizers.l2(5e-4))(fc)
    model = ks.Model(inputs=inputs,
                     outputs=[predict_face, predict_bbox, predict_landmark])
    return model


def o_net():
    """
    :return: o_net model
    """
    inputs = ks.Input((o_net_size, o_net_size, 3))
    x = kl.Conv2D(32, 3, activation=prelu, kernel_regularizer=ks.regularizers.l2(5e-4))(inputs)
    x = kl.MaxPool2D((3, 3), 2, 'same')(x)
    x = kl.Conv2D(64, 3, activation=prelu, kernel_regularizer=ks.regularizers.l2(5e-4))(x)
    x = kl.MaxPool2D((3, 3), 2, 'valid')(x)
    x = kl.Conv2D(64, 3, activation=prelu, kernel_regularizer=ks.regularizers.l2(5e-4))(x)
    x = kl.MaxPool2D((2, 2), 2, 'same')(x)
    x = kl.Conv2D(128, 2, activation=prelu, kernel_regularizer=ks.regularizers.l2(5e-4))(x)
    flatten = kl.Flatten()(x)
    fc = kl.Dense(256, activation=prelu, kernel_regularizer=ks.regularizers.l2(5e-4))(flatten)

    predict_face = kl.Dense(2, activation=tf.nn.softmax, kernel_regularizer=ks.regularizers.l2(5e-4))(fc)
    predict_bbox = kl.Dense(4, kernel_regularizer=ks.regularizers.l2(5e-4))(fc)
    predict_landmark = kl.Dense(10, kernel_regularizer=ks.regularizers.l2(5e-4))(fc)
    model = ks.Model(inputs=inputs,
                     outputs=[predict_face, predict_bbox, predict_landmark])
    return model


def prelu(inputs):
    """
    prelu activation function
    :param inputs:
    :return:
    """
    alphas = tf.constant(0.25, dtype=tf.float32)
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs - abs(inputs)) * 0.5
    return pos + neg


def cls_ohem(cls_prob, label):
    """
    loss in classification
    :param cls_prob: class probability
    :param label:
    :return:
    """
    if len(cls_prob.shape) == 4:
        cls_prob = tf.squeeze(cls_prob, [1, 2])
        # print('cls_prob = {}'.format(cls_prob.numpy()))
    zeros = tf.zeros_like(label)
    # 只把pos的label设定为1,其余都为0
    label_filter_invalid = tf.where(tf.less(label, 0), zeros, label)
    # 类别size[2*batch]
    num_cls_prob = tf.size(cls_prob)
    cls_prob_reshpae = tf.reshape(cls_prob, [num_cls_prob, -1])
    label_int = tf.cast(label_filter_invalid, tf.int32)
    # 获取batch数
    num_row = tf.cast(cls_prob.get_shape()[0], tf.int32)
    # 对应某一batch而言，第batch*2行是neg的概率，batch*2+1行是pos的概率,indices为对应 cls_prob_reshpae
    # 应该的真实值，后续用交叉熵计算损失
    row = tf.range(num_row) * 2
    indices_ = row + label_int
    # 真实标签对应的概率
    label_prob = tf.squeeze(tf.gather(cls_prob_reshpae, indices_))
    loss = -tf.math.log(label_prob + 1e-10)
    zeros = tf.zeros_like(label_prob, dtype=tf.float32)
    ones = tf.ones_like(label_prob, dtype=tf.float32)
    # 统计neg和pos的数量
    valid_inds = tf.where(label < zeros, zeros, ones)
    num_valid = tf.reduce_sum(valid_inds)
    # 选取70%的数据
    keep_num = tf.cast(num_valid * num_keep_ratio, dtype=tf.int32)
    # 只选取neg，pos的70%损失
    loss = loss * valid_inds
    loss, _ = tf.nn.top_k(loss, k=keep_num)
    return tf.reduce_mean(loss)


def bbox_ohem(bbox_pred, bbox_target, label):
    """
    loss of bounding box
    :param bbox_pred:
    :param bbox_target:
    :param label:
    :return:
    """
    if len(bbox_pred.shape) == 4:
        bbox_pred = tf.squeeze(bbox_pred, [1, 2])
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    ones_index = tf.ones_like(label, dtype=tf.float32)
    # 保留pos和part的数据
    valid_inds = tf.where(tf.equal(tf.abs(label), 1), ones_index, zeros_index)
    # 计算平方差损失
    square_error = tf.square(bbox_pred - bbox_target)
    square_error = tf.reduce_sum(square_error, axis=1)
    # 保留的数据的个数
    num_valid = tf.reduce_sum(valid_inds)
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    # 保留pos和part部分的损失
    square_error = square_error * valid_inds
    # square_error, _ = tf.nn.top_k(square_error, k=keep_num)
    return tf.reduce_mean(square_error)


def landmark_ohem(landmark_pred, landmark_target, label):
    """
    loss of landmarks
    :param landmark_pred:
    :param landmark_target:
    :param label:
    :return:
    """
    if len(landmark_pred.shape) == 4:
        landmark_pred = tf.squeeze(landmark_pred, [1, 2])
    ones = tf.ones_like(label, dtype=tf.float32)
    zeros = tf.zeros_like(label, dtype=tf.float32)
    # 只保留landmark数据
    valid_inds = tf.where(tf.equal(label, -2), ones, zeros)
    # 计算平方差损失
    square_error = tf.square(landmark_pred - landmark_target)
    square_error = tf.reduce_sum(square_error, axis=1)
    # 保留数据个数
    num_valid = tf.reduce_sum(valid_inds)
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    # 保留landmark部分数据损失
    # print('square error = {}, valid_inds = {}'.format(square_error.numpy(), valid_inds.numpy()))
    square_error = square_error * valid_inds
    # square_error, _ = tf.nn.top_k(square_error, k=keep_num)
    return tf.reduce_mean(square_error)


def cal_accuracy(cls_prob, label):
    """
    calculate accuracy
    :param cls_prob:
    :param label:
    :return:
    """
    if len(cls_prob.shape) == 4:
        cls_prob = tf.squeeze(cls_prob, [1, 2])
    # 预测最大概率的类别，0代表无人，1代表有人
    pred = tf.argmax(cls_prob, axis=1)
    label_int = tf.cast(label, tf.int64)
    # 保留label>=0的数据，即pos和neg的数据
    cond = tf.where(tf.greater_equal(label_int, 0))
    picked = tf.squeeze(cond)
    # 获取pos和neg的label值
    label_picked = tf.gather(label_int, picked)
    pred_picked = tf.gather(pred, picked)
    # print('label picked = {}, pred_picked={}'.format(label_picked.numpy(),
    #                                                  pred_picked.numpy()))
    # 计算准确率
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(label_picked, pred_picked), tf.float32))
    return accuracy_op
