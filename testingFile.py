import numpy as np
from scipy.misc import imread
from scipy.misc import imsave
import os
import foolbox
import tensorflow as tf


import glob
import re

def load_filenames_labels(mode):
    #https://github.com/pat-coady/tiny_imagenet/blob/master/src/input_pipe.py
    label_dict, class_description = build_label_dicts()
    filenames_labels = []
    if mode == 'train':
        filenames = glob.glob('./tiny-imagenet-200/train/*/images/*.JPEG')
        for filename in filenames:
            match = re.search(r'n\d+', filename)
            label = str(label_dict[match.group()])
            filenames_labels.append((filename, label))
    elif mode == 'val':
        with open('./tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
            for line in f.readlines():
                split_line = line.split('\t')
                filename = '../tiny-imagenet-200/val/images/' + split_line[0]
                label = str(label_dict[split_line[1]])
    filenames_labels.append((filename, label))

    return filenames_labels


def build_label_dicts():
    """
    label_dict:
    keys = synset(e.g."n01944390")
    values = class integer {0..199}
    class_desc:
    keys = class integer {0..199}
    values = text description from words.txt
    """
    label_dict, class_description = {}, {}
    with open('./tiny-imagenet-200/wnids.txt', 'r') as f:
        for i, line in enumerate(f.readlines()):
            synset = line[:-1]  # remove \n
            label_dict[synset] = i
    with open('./tiny-imagenet-200/words.txt', 'r') as f:
        for i, line in enumerate(f.readlines()):
            synset, desc = line.split('\t')
            desc = desc[:-1]  # remove \n
            if synset in label_dict:
                class_description[label_dict[synset]] = desc

    return label_dict, class_description


def read_images(batch_size, image_type):
    #image_type: train, test, val

    print("hello")
    image_dir = "./tiny-imagenet-200/" + image_type
    images = np.zeros((batch_size, 64, 64, 3))
    filenames = []
    labels = []
    idx = 0
    print(image_dir)
    from PIL import Image

    print(os.listdir(image_dir))
    for type in os.listdir(image_dir):
        dir = image_dir + '/' + type + '/images/'
        if os.path.isdir(dir):
            for filepath in tf.gfile.Glob(os.path.join(dir, '*.JPEG')):
                with tf.gfile.Open(filepath) as f:
                    image = imread(filepath).astype(np.float)
                img[1,2,1] = img[2,3,1]
                images[idx, :, :, :] = image
                filenames.append(os.path.basename(filepath))
                labels.append(type)
                print(np.array(image).shape)
                idx += 1
                if idx == batch_size:
                    return filenames, images, labels
                    filenames = []
                    images = np.zeros(batch_size, 64, 64, 3)
                    idx = 0
            if idx > 0:
                return filenames, images, labels


def test():
    img = tf.placeholder(tf.float32,(None,2,2))

    input = tf.random_uniform([2,2],0,1,dtype=tf.int32)

    img[1,1] = img[1,2]



    tf.Session().run([img],input)

    # a = tf.constant([[1, 2, 3], [5, 6, 7]])
    # #a[1][1] = a[0][1]
    # b = tf.gather(a,[0])
    # b_ = tf.Session().run(b)
    # print(b_[0][0])
    # ind = [[1,1]]
    # value = b_[0][0]
    # sh = (2,3)
    # delta = tf.SparseTensor(ind,value,sh)
    # a = tf.sparse_tensor_to_dense(delta)
    # print(tf.Session().run(a))



def main():
    test()
    # batch_size = 10
    # image_type = 'train'
    # filenames, images, labels = read_images(batch_size, image_type)
    # #filenames_labels = load_filenames_labels('train')
    # #label_dict, class_description = build_label_dicts()
    # print('yes')

if __name__ == '__main__':
    #pass_model_to_server()
    main()

