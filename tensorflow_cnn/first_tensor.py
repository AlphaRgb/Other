#!/usr/bin/env python3
# coding=utf-8

import tensorflow as tf

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# hello = tf.constant('hello ,world')
# sess = tf.Session()
# print(sess.run(hello))

# a = tf.constant(2)
# b = tf.constant(3)
# sess = tf.Session()
# print(sess.run(a+b))
# print(sess.run(a*b))

# a = tf.placeholder(tf.int16)
# b = tf.placeholder(tf.int16)

# add = tf.add(a,b)
# # mul = tf.mul(a,b)
# sess = tf.Session()
# print(sess.run(add,feed_dict={a:3,b:4}))

# from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
# print(mnist.train.images.shape,mnist.train.labels.shape)
# print(mnist.test.images.shape,mnist.test.labels.shape)
# print(mnist.validation.images.shape,mnist.validation.labels.shape)

# matrix1 = tf.constant([[3.,3.]])
# matrix2 = tf.constant([[2.],[2.]])

# product = tf.matmul(matrix1,matrix2)

# print(matrix1)
# print(matrix2)
# print(product)

# # sess = tf.Session()
# # result = sess.run(product)

# # print(result)

# with tf.Session() as sess:
# 	result = sess.run(product)
# 	print(result)

# sess.close()

sess = tf.InteractiveSession()

x = tf.Variable([1.0,2.0])
print(x)
a = tf.constant([3.0,3.0])
print(a)

x.initializer.run()
print(x)

sub = tf.subtract(x,a)
print(sub.eval())













