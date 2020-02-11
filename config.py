# coding=utf-8

p_net_size = 12
r_net_size = 24
o_net_size = 48

# min face size
min_face = 20

# p_net input/output scale
p_net_stride = 2
# thresholds for three nets
face_thresholds = [0.6, 0.7, 0.7]
# last mode for test
test_mode = 'o_net'
# 1 for image, 2 for camera
input_mode = '1'
# dir for test image
test_dir = 'picture/'
# output dir for test image
out_path = 'output/'
