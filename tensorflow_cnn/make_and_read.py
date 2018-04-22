#!/usr/bin/env python3
# coding=utf-8

# import tensorflow as tf
# import os
#
# # 制作
# def _int64_feature(value):
#   return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#
#
# def _bytes_feature(value):
#   return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
#
# # images and labels array as input
# def convert_to(images, labels, name):
#   num_examples = labels.shape[0]
#   if images.shape[0] != num_examples:
#     raise ValueError("Images size %d does not match label size %d." %
#                      (images.shape[0], num_examples))
#   rows = images.shape[1]
#   cols = images.shape[2]
#   depth = images.shape[3]
#
#   filename = os.path.join(FLAGS.directory, name + '.tfrecords')
#   print('Writing', filename)
#   writer = tf.python_io.TFRecordWriter(filename)
#   for index in range(num_examples):
#     image_raw = images[index].tostring()
#     example = tf.train.Example(features=tf.train.Features(feature={
#         'height': _int64_feature(rows),
#         'width': _int64_feature(cols),
#         'depth': _int64_feature(depth),
#         'label': _int64_feature(int(labels[index])),
#         'image_raw': _bytes_feature(image_raw)}))
#     writer.write(example.SerializeToString())
#
# # 读取
# # Remember to generate a file name queue of you 'train.TFRecord' file path
# def read_and_decode(filename_queue):
#   reader = tf.TFRecordReader()
#   _, serialized_example = reader.read(filename_queue)
#   features = tf.parse_single_example(
#     serialized_example,
#     dense_keys=['image_raw', 'label'],
#     # Defaults are not specified since both keys are required.
#     dense_types=[tf.string, tf.int64])
#
#   # Convert from a scalar string tensor (whose single string has
#   image = tf.decode_raw(features['image_raw'], tf.uint8)
#
#   image = tf.reshape(image, [my_cifar.n_input])
#   image.set_shape([my_cifar.n_input])
#
#   # OPTIONAL: Could reshape into a 28x28 image and apply distortions
#   # here.  Since we are not applying any distortions in this
#   # example, and the next step expects the image to be flattened
#   # into a vector, we don't bother.
#
#   # Convert from [0, 255] -> [-0.5, 0.5] floats.
#   image = tf.cast(image, tf.float32)
#   image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
#
#   # Convert label from a scalar uint8 tensor to an int32 scalar.
#   label = tf.cast(features['label'], tf.int32)
#
#   return image, label

import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

## 写入tfrecord文件
cwd = '/Users/alpha/Ai/images/'
classes = {'cat','dog'}
writer = tf.python_io.TFRecordWriter('animal_train.tfrecords')

for index,name in enumerate(classes):
    class_path = cwd + name + '/'
    print(os.listdir(class_path))
    for img_name in os.listdir(class_path):
        img_path = class_path + img_name

        img = Image.open(img_path)
        img = img.resize((64,64))
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),'img_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))}))
        writer.write(example.SerializeToString())
writer.close()


## 读取tfrecord文件
# def read_and_decode(filename):
# filename_queue = tf.train.string_input_producer(['animal_train.tfrecords'])
# reader = tf.TFRecordReader()
# _,serialized_example = reader.read(filename_queue)
# features = tf.parse_single_example(serialized_example,features={'label':tf.FixedLenFeature([],tf.int64),'img_raw':tf.FixedLenFeature([],tf.string)})
# img = tf.decode_raw(features['img_raw'],tf.uint8)
# img = tf.reshape(img,[128,128,3])
# # img = tf.cast(img,tf.float32)*(1./255) - 0.5
# label = tf.cast(features['label'],tf.int32)
#     # return img,label
#
# with tf.Session() as sess:
#     init_op = tf.initialize_all_variables()
#     sess.run(init_op)
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#     for i in range(6):
#         example,l = sess.run([img,label])
#         img = Image.fromarray(example,'RGB')
#         img.save(cwd+str(i)+'_lable_'+str(l)+'.jpg')
#         print(example,l)
#     coord.request_stop()
#     coord.join(threads)




