# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensorflow.contrib.slim.nets import resnet_v2
import tensorflow as tf
from foolbox.models import TensorFlowModel
import numpy as np


def create_model():
    model_name = 'resnet_v2_50'
    checkpoint_path = "tiny_imagenet_alp_checkpoints/tiny_imagenet_alp05_2018_06_26.ckpt"
    num_classes = 200

    graph = tf.Graph()
    with graph.as_default():
        images = tf.placeholder(tf.float32, (None, 64, 64, 3))
        # setup model
        #input = np.random.uniform(-1,1,(5,64,64,3))
        with tf.contrib.framework.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, _ = resnet_v2.resnet_v2_50(images, num_classes, is_training=False, reuse=tf.AUTO_REUSE)
            logits = tf.reshape(logits, [-1, num_classes])

        variables_to_restore = tf.contrib.framework.get_variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

    sess = tf.Session(graph=graph)
    saver.restore(sess, checkpoint_path)

    with sess.as_default():
        #sess.run(tf.global_variables_initializer())
        #log_comp = sess.run([logits],feed_dict={images:input})
        model = TensorFlowModel(images, logits, bounds=(0,255))
        return model
    #return log_comp


if __name__ == '__main__':
    #app.run(main)
    print(create_model())

