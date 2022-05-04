import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

test_feat = [[0.1,0.2,0.3,0.4,0.5,0.6],[0.6,0.5,0.4,0.3,0.2,0.1], [0.2,0.3,0.4,0.5,0.6,0.7], [0.7,0.6,0.5,0.4,0.3,0.2], [0.5,0.6,0.7,0.8,0.9,0.1], [0.8,0.7,0.6,0.5,0.4,0.3]]
test_lbl = [[1], [2], [1], [2], [1], [2]]

EPOCHS = 10
BATCH_SIZE = 16
print()

features, labels = (test_feat, test_lbl)
print(features)
print(labels)

dataset = tf.data.Dataset.from_tensor_slices((features,labels)).repeat().batch(BATCH_SIZE)
iter = dataset.make_one_shot_iterator()
x, y = iter.get_next()

net = tf.layers.dense(x, 8, activation=tf.tanh)
net = tf.layers.dense(net, 8, activation=tf.tanh)
prediction = tf.layers.dense(net, 1, activation=tf.tanh)
loss = tf.losses.mean_squared_error(prediction, y)
train_op = tf.train.AdamOptimizer().minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(EPOCHS):
        _, loss_value = sess.run([train_op, loss])
        print("Iter: {}, Loss: {:.4f}".format(i, loss_value))
