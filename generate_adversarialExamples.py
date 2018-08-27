from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags

import tensorflow as tf

import adversarial_attack
import model_lib
from datasets import dataset_factory

FLAGS = flags.FLAGS

num_examples_per_epoch = 4000


def tiny_imagenet_parser(value, image_size, is_training):
    """Parses tiny imagenet example.

    Args:
    value: encoded example.
    image_size: size of the image.
    is_training: if True then do training preprocessing (which includes
      random cropping), otherwise do eval preprocessing.

    Returns:
    image: tensor with the image.
    label: true label of the image.
    """
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, ''),
        'label/tiny_imagenet': tf.FixedLenFeature([], tf.int64, -1),
    }

    parsed = tf.parse_single_example(value, keys_to_features)

    image_buffer = tf.reshape(parsed['image/encoded'], shape=[])
    image = tf.image.decode_image(image_buffer, channels=3)
    image = tf.image.convert_image_dtype(
        image, dtype=tf.float32)

    # Crop image
    if is_training:
        bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=tf.constant([0.0, 0.0, 1.0, 1.0],
                                       dtype=tf.float32,
                                       shape=[1, 1, 4]),
            min_object_covered=0.5,
            aspect_ratio_range=[0.75, 1.33],
            area_range=[0.5, 1.0],
            max_attempts=20,
            use_image_if_no_bounding_boxes=True)
        image = tf.slice(image, bbox_begin, bbox_size)

    # resize image
    image = tf.image.resize_bicubic([image], [image_size, image_size])[0]

    # Rescale image to [-1, 1] range.
    image = tf.multiply(tf.subtract(image, 0.5), 2.0)

    image = tf.reshape(image, [image_size, image_size, 3])

    # Labels are in [0, 199] range
    label = tf.cast(
        tf.reshape(parsed['label/tiny_imagenet'], shape=[]), dtype=tf.int32)

    print('here')
    return image, label


def main():
    filepath = os.path.join('tiny-imagenet-tfrecord', 'train.tfrecord')
    dataset = tf.data.TFRecordDataset(filepath, buffer_size=8 * 1024 * 1024)
    image_size = 64
    is_training = False
    #dataset = tf.map_fn(lambda value: tiny_imagenet_parser(value, image_size, is_training), dataset)
    dataset = dataset.map(lambda value: tiny_imagenet_parser(value, image_size, is_training))
    # dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    num_examples = 4000
    num_classes = 200
    bounds = (-1, 1)
    dataset_iterator = dataset.make_one_shot_iterator()
    image, label = dataset_iterator.get_next()
    one_hot_label = tf.one_hot(label, num_classes)

    with tf.Session() as sess:
        print(sess.run([image, label]))

if __name__ == '__main__':
    main()