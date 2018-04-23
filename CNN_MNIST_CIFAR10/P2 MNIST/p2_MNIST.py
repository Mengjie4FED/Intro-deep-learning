import tensorflow as tf

# input data directly fron tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

sess = tf.InteractiveSession()

# define functions for generate weights and bias
def weight(s):
    w = tf.truncated_normal(s, stddev = 0.05)
    return tf.Variable(w)

def bias(s):
    b = tf.constant(0.05, shape = s)
    return tf.Variable(b)

# define functions for convolution and pooling
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 1st: filter size 3*3; 64 nodes;
W_conv1  = weight([3, 3, 1, 64])  
b_conv1 = bias([64])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 2nd: filter size 3*3; 128 nodes;
W_conv2 = weight([3, 3, 64, 128])
b_conv2 = bias([128])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# connected
W_fc1 = weight([7 * 7 * 128, 1024])
b_fc1 = bias([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout: the probability that a neuron's output is kept
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout
W_fc2 = weight([1024, 10])
b_fc2 = bias([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# train and evaluate
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  # 10,000 training iterations; 100 batch size; 0.5 keep probability
  for i in range(10000): 
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

  print('test accuracy %g with keep probability of 0.5' % accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  # 10,000 training iterations; 100 batch size; without dropout
  for i in range(10000): 
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1})

  print('test accuracy %g without dropout' % accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
 