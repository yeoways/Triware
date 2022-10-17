# Copyright 2020 University of Illinois Board of Trustees. All Rights Reserved.
# Author: Beomyeol Jeon, DPRG (https://dprg.cs.uiuc.edu)
# This file is part of Baechi, which is released under specific terms. See file License.txt file for full license details.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf

from cv.networks import cifarnet
from cv.networks import inception_v3
from cv.networks import resnet_v1
from cv.networks.nasnet import nasnet
from cv.networks import vgg

slim = tf.contrib.slim

networks_generator = {
    'cifarnet': cifarnet.cifarnet,
    'inception_v3': inception_v3.inception_v3,
    'resnet_v1_101': resnet_v1.resnet_v1_101,
    'resnet_v1_50': resnet_v1.resnet_v1_50,
    'resnet_v1_152': resnet_v1.resnet_v1_152,
    'resnet_v1_200': resnet_v1.resnet_v1_200,
    'nasnet_cifar': nasnet.build_nasnet_cifar,
    'vgg_a': vgg.vgg_a,
    'vgg_16': vgg.vgg_16,
    'vgg_19': vgg.vgg_19,

}

arg_scopes_generator = {
    'cifarnet': cifarnet.cifarnet_arg_scope,
    'inception_v3': inception_v3.inception_v3_arg_scope,
    'resnet_v1_101': resnet_v1.resnet_arg_scope,
    'resnet_v1_50': resnet_v1.resnet_arg_scope,
    'resnet_v1_152': resnet_v1.resnet_arg_scope,
    'resnet_v1_200': resnet_v1.resnet_arg_scope,
    'nasnet_cifar': nasnet.nasnet_cifar_arg_scope,
    'vgg_a': vgg.vgg_arg_scope,
    'vgg_16': vgg.vgg_arg_scope,
    'vgg_19': vgg.vgg_arg_scope,
}


def get_network_fn(name, num_classes, weight_decay=0.0, is_training=False):
  if name not in networks_generator:
    raise ValueError('Name of network unknown %s' % name)
  func = networks_generator[name]

  @functools.wraps(func)
  def network_fn(images, **kwargs):
    arg_scope = arg_scopes_generator[name](weight_decay=weight_decay)
    with slim.arg_scope(arg_scope):
      return func(images, num_classes=num_classes, is_training=is_training,
                  **kwargs)
  if hasattr(func, 'default_image_size'):
    network_fn.default_image_size = func.default_image_size

  return network_fn
