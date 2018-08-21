

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from train import *

def main(_):
  assert FLAGS.output_dir, '--output_dir has to be provided'
  if not tf.gfile.Exists(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)
  params = model_lib.default_hparams()
  params.parse(FLAGS.hparams)
  tf.logging.info('User provided hparams: %s', FLAGS.hparams)
  tf.logging.info('All hyper parameters: %s', params)
  batch_size = params.batch_size
  graph = tf.Graph()
  with graph.as_default():
    with tf.device(tf.train.replica_device_setter(ps_tasks=FLAGS.ps_tasks)):
      # dataset
      dataset, examples_per_epoch, num_classes, bounds = (
          dataset_factory.get_dataset(
              FLAGS.dataset,
              'train',
              batch_size,
              FLAGS.dataset_image_size,
              is_training=True))
      dataset_iterator = dataset.make_one_shot_iterator()
      images, labels = dataset_iterator.get_next()
      one_hot_labels = tf.one_hot(labels, num_classes)

      # set up model
      global_step = tf.train.get_or_create_global_step()
      model_fn = model_lib.get_model(FLAGS.model_name, num_classes)

      model_fn_eval_mode = lambda x: model_fn(x, is_training=False)
      adv_examples = adversarial_attack.generate_adversarial_examples(
            images, bounds, model_fn_eval_mode, params.train_adv_method)
      all_examples = tf.concat([images, adv_examples], axis=0)
      logits = model_fn(all_examples, is_training=True)
      one_hot_labels = tf.concat([one_hot_labels, one_hot_labels], axis=0)
  return