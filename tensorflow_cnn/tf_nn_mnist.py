#!/usr/bin/env python3
# coding=utf-8

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math

mnist = input_data.read_data_sets('./mnist/',one_hot=True)

x_train = mnist.train.images
y_train = mnist.train.labels
x_dev = mnist.validation.images
y_dev = mnist.validation.labels
x_test = mnist.test.images
y_test = mnist.test.labels

print(x_train)
print(x_train.T.shape)
print(y_train.T.shape)

def display_digit(index):
    print(y_train[index])
    label = y_train[index].argmax(axis=0)
    image = x_train[index].reshape([28,28])
    plt.title('example:%d label:%d'%(index,label))
    plt.imshow(image,cmap=plt.get_cmap('gray_r'))
    plt.show()
    pass

# display_digit(5)
# print(y_train[5].shape)

# 样本转置
x_train = x_train.T
y_train = y_train.T
x_dev = y_dev.T
y_dev = y_dev.T
x_test = x_test.T
y_test = y_test.T

def random_mini_batches(X,Y,mini_batch_size=64):
    m = X.shape[1]
    mini_batches = []
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:,permutation]
    shuffled_Y = Y[:,permutation].reshape((-1,m))

    num_complete_minibatches = math.floor(m/mini_batch_size)

    for k in range(0,num_complete_minibatches):
        mini_batch_X = shuffled_X[:,k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k*mini_batch_size:(k+1)*mini_batch_size]

        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size !=0:
        mini_batch_X = shuffled_X[:,mini_batch_size*num_complete_minibatches:]
        mini_batch_Y = shuffled_Y[:,mini_batch_size*num_complete_minibatches:]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

# 参数初始化
layer_dims = [784,64,128,10]

def init_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims) - 1
    for l in range(1,L+1):
        parameters['W'+str(l)] = tf.Variable(tf.random_normal([layer_dims[l],layer_dims[l-1]]))
        parameters['b'+str(l)] = tf.Variable(tf.random_normal([layer_dims[l],1]))
    return parameters

def forward_propagation(X,parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1,X),b1)  #Z1 = np.dot(W1,X) + b1
    A1 = tf.nn.relu(Z1)              #A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)#Z2 = np.dot(W1,A1) + b2
    A2 = tf.nn.relu(Z2)              #A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)#Z3 = np.dot(W1,A2) + b1
    return Z3

def compute_cost(Z3,Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))
    return cost

def tf_nn_model(X_train,Y_train,X_test,Y_test,layer_dims,learning_rate=0.001,num_epochs=100,minibatch_size=64,print_cost=True):
    n_x,m = X_train.shape
    n_y = Y_train.shape[0]
    costs = []
    X = tf.placeholder(tf.float32,[n_x,None],name='X')
    Y = tf.placeholder(tf.float32,[n_y,None],name='Y')

    parameters = init_parameters(layer_dims)
    Z3 = forward_propagation(X,parameters)

    cost = compute_cost(Z3,Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0
            num_minibatches = int(m/minibatch_size)
            minibatches = random_mini_batches(X_train,Y_train,minibatch_size)
            for minibatche in minibatches:
                minibatch_X ,minibatch_Y = minibatche
                _,minibatch_cost = sess.run([optimizer,cost],feed_dict={X:minibatch_X,Y:minibatch_Y})
                epoch_cost += minibatch_cost/num_minibatches
            if print_cost == True and epoch % 10 == 0:
                print('cost after epoch %i:%f'%(epoch,epoch_cost))
            if print_cost == True and epoch%5 == 0:
                costs.append(epoch_cost)
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title('learning rate='+str(learning_rate))
        plt.show()

        parameters = sess.run(parameters)

        correct_prediction = tf.equal(tf.argmax(Z3),tf.argmax(Y))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        return parameters

tf_nn_model(x_train,y_train,x_test,y_test,layer_dims,learning_rate=0.001,num_epochs=100,minibatch_size=64,print_cost=True)