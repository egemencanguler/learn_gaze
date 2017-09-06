import tensorflow as tf
import json


with open('results.json') as json_file:
    data = json.load(json_file)

features = data["features"]
points = data["labels"]


features = [[x] for x in range(100)]
points = [[x**2,x*2] for x in range(100)]

n_samples = len(features)
n_input = 1
n_out = 2
n_hidden_layer = 1

X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_out])

W1 = tf.Variable(tf.zeros([n_input,n_hidden_layer]))
b1 = tf.Variable(tf.zeros([n_hidden_layer]))

W2 = tf.Variable(tf.zeros([n_hidden_layer,n_out]))
b2 = tf.Variable(tf.zeros(n_out))

hidden_layer = tf.matmul(X,W1) + b1
pred = tf.matmul(tf.nn.relu(hidden_layer),W2) + b2

# Loss function using L2 Regularization
beta = 0.1
regularizer = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)
# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
loss = tf.reduce_mean(cost)

optimizer = tf.train.AdadeltaOptimizer()
train_op = optimizer.minimize(loss)


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(10000):
        sess.run(train_op, feed_dict={X:features,Y:points})
        c = sess.run(loss, feed_dict={X:features,Y:points})
        print("Cost",c/n_samples)














