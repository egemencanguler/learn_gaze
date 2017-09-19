import tensorflow as tf
import numpy as np
from gaze_data import GazeData

# Prepare the data
test_file = "webgaze_results/bahadır.json"
gaze_data = GazeData(test_file)

features = np.array(gaze_data.cal_features)
pointsX = np.array([x[0] for x in gaze_data.cal_points])
pointsY = np.array([x[1] for x in gaze_data.cal_points])


n_test = int(len(features) / 4)
n_train = len(features) - n_test

idx = np.array(list(range(len(features))))
np.random.shuffle(idx)
train_features = features[idx[:n_train]]
train_x = pointsX[idx[:n_train]].reshape((n_train,1))
train_y = pointsY[idx[:n_train]]


test_features = features[idx[n_train:]]
test_x = pointsX[idx[n_train:]].reshape((n_test,1))
test_y = pointsY[idx[n_train:]]

# Train Model


n_samples = len(train_features)
n_input = 120
n_out = 1
n_hidden_layer = 5

X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_out])

W1 = tf.Variable(tf.zeros([n_input,n_hidden_layer]))
b1 = tf.Variable(tf.zeros([n_hidden_layer]))

W2 = tf.Variable(tf.zeros([n_hidden_layer,n_out]))
b2 = tf.Variable(tf.zeros(n_out))

hidden_layer = tf.matmul(X,W1) + b1
pred = tf.matmul(tf.nn.relu(hidden_layer),W2) + b2

# Loss function using L2 Regularization
beta = 0.05
regularizer = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)
# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
loss = tf.reduce_mean(cost + beta * regularizer)

optimizer = tf.train.GradientDescentOptimizer(0.0001)
train_op = optimizer.minimize(loss)


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(1110000):
        sess.run(train_op, feed_dict={X:train_features,Y:train_x})
        c = sess.run(loss, feed_dict={X:train_features,Y:train_x})
        print("Cost",c/n_samples)














