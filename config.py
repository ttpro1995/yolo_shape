# -*- coding: utf-8 -*-

#  grid system
cell_size = 7
#  boundary box
box_per_cell = 2
#
img_size = 224
#
classes = {'circle': 0, 'triangle': 1, 'rectangle': 2}
nclass = len(classes)

box_scale = 5.0
noobject_scale = 0.5
batch_size = 128
#
#
epochs = 1000
# learning rate
lr = 1e-3