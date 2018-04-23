#!/usr/bin/env python3
# coding=utf-8


import numpy as np
import tensorflow as tf


# 生成一个[0,1)之间的随机浮点数或N维浮点数组
x_data = np.float32(np.random.rand(2, 100))
# 第一个矩阵中与该元素行号相同的元素与第二个矩阵与该元素列号相同的元素，两两相乘后再求和
y_data = np.dot([0.100, 0.200], x_data) + 0.300
print(x_data)
print(y_data)

# 构造一个线性模型
b = tf.Variable(tf.zeros([1]))
# 返回1*2的矩阵，产生于minval和maxval之间，产生的值是均匀分布的
W = tf.Variable(tf.random_uniform([1,2], -1, 1.0))
# 矩阵乘法
y = tf.matmul(W, x_data) + b
print(y)


# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
optimize = tf.train.GradientDescentOptimizer(0.5)
train = optimize.minimize(loss)


# 初始化变量
init = tf.global_variables_initializer()

# # 启动图
# sess = tf.Session()
# sess.run(init)


# # 拟合平面
# for step in range(0, 201):
#     sess.run(train)
#     if step % 20 == 0:
#         print(step , sess.run(W), sess.run(b))


# # 在一个会话中启动图
# # 创建常量op
# matrix1 = tf.constant([[3.,3.]])
# print(matrix1)
#
#
# matrix2 = tf.constant([[2.], [2.]])
# print(matrix1)
#
#
# product = tf.matmul(matrix1, matrix2)
#
#
# with tf.Session() as sess:
#     with tf.device('/gpu:0'):
#         sess.run([product])
#         print(product)


# 交互式使用
sess = tf.InteractiveSession()


x = tf.Variable([1.0, 2.0])
a = tf.Variable([3.0, 3.0])


x.initializer.run()

# 增加一个减法op
sub = tf.subtract(x, a)
print(sub)







