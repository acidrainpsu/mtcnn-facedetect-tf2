# coding = utf-8
import os
import numpy as np

data_dir = '../data/'

"""将pos,part,neg,landmark四者混在一起"""
with open(os.path.join(data_dir, 'p_net/p_net_pos.txt'), 'r') as f:
    pos = f.readlines()
with open(os.path.join(data_dir, 'p_net/p_net_neg.txt'), 'r') as f:
    neg = f.readlines()
with open(os.path.join(data_dir,'p_net/p_net_part.txt'), 'r') as f:
    part = f.readlines()
with open(os.path.join(data_dir, 'p_net/landmark_p_net_aug.txt'), 'r') as f:
    landmark = f.readlines()
dir_path = os.path.join(data_dir, 'p_net')
if not os.path.exists(dir_path):
    os.mkdir(dir_path)
with open(os.path.join(dir_path, 'train_p_net_landmark.txt'), 'w') as f:
    nums = [len(neg), len(pos), len(part)]
    base_num = 250000
    print('neg num：{}, pos num：{}, part num:{} base:{}'.format(len(neg), len(pos), len(part), base_num))
    if len(neg) > base_num*3:
        neg_keep = np.random.choice(len(neg), size=base_num*3, replace=True)
    else:
        neg_keep = np.random.choice(len(neg), size=len(neg), replace=True)
    sum_p = len(neg_keep)//3
    pos_keep = np.random.choice(len(pos), sum_p, replace=True)
    part_keep = np.random.choice(len(part), sum_p, replace=True)
    print('neg num：{} pos num：{} part num:{}'.format(len(neg_keep), len(pos_keep), len(part_keep)))
    for i in pos_keep:
        f.write(pos[i])
    for i in neg_keep:
        f.write(neg[i])
    for i in part_keep:
        f.write(part[i])
    for item in landmark:
        f.write(item)

