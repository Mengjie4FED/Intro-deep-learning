import tensorflow as tf

def conv(x, w, b, name):
    with tf.variable_scope('conv'):
        return tf.nn.conv2d(x,
                           filter=w,
                           strides=[1, 1, 1, 1],
                           padding='SAME',
                           name=name) + b

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def cifar10_conv_L8(X, keep_prob, reuse=False):
    with tf.variable_scope('cifar10_conv_L8'):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        batch_size = tf.shape(X)[0]
        K = 64
        M = 128
        N = 256
        P = 300
        Q = 300
        R = 300
        S = 300
        T = 300

        W1 = tf.get_variable('D_W1', [5, 5, 3, K], initializer=tf.contrib.layers.xavier_initializer())
        B1 = tf.get_variable('D_B1', [K], initializer=tf.constant_initializer())

        W2 = tf.get_variable('D_W2', [5, 5, K, M], initializer=tf.contrib.layers.xavier_initializer())
        B2 = tf.get_variable('D_B2', [M], initializer=tf.constant_initializer())

        W3 = tf.get_variable('D_W3', [5, 5, M, N], initializer=tf.contrib.layers.xavier_initializer())
        B3 = tf.get_variable('D_B3', [N], initializer=tf.constant_initializer())

        W4 = tf.get_variable('D_W4', [5, 5, N, P], initializer=tf.contrib.layers.xavier_initializer())
        B4 = tf.get_variable('D_B4', [P], initializer=tf.constant_initializer())

        W5 = tf.get_variable('D_W5', [5, 5, P, Q], initializer=tf.contrib.layers.xavier_initializer())
        B5 = tf.get_variable('D_B5', [Q], initializer=tf.constant_initializer())

        W6 = tf.get_variable('D_W6', [5, 5, Q, R], initializer=tf.contrib.layers.xavier_initializer())
        B6 = tf.get_variable('D_B6', [R], initializer=tf.constant_initializer())

        W7 = tf.get_variable('D_W7', [5, 5, R, S], initializer=tf.contrib.layers.xavier_initializer())
        B7 = tf.get_variable('D_B7', [S], initializer=tf.constant_initializer())

        W8 = tf.get_variable('D_W8', [5, 5, S, T], initializer=tf.contrib.layers.xavier_initializer())
        B8 = tf.get_variable('D_B8', [T], initializer=tf.constant_initializer())

        W9 = tf.get_variable('D_W9', [12*12*T, 10], initializer=tf.contrib.layers.xavier_initializer())
        B9 = tf.get_variable('D_B9', [10], initializer=tf.constant_initializer())

        conv1 = conv(X, W1, B1, name='conv1')
        bn1 = tf.nn.relu(tf.contrib.layers.batch_norm(conv1))

        conv2 = conv(bn1, W2, B2, name='conv2')
        bn2 = tf.nn.relu(tf.contrib.layers.batch_norm(conv2))

        conv3 = conv(bn2, W3, B3, name='conv3')
        bn3 = tf.nn.relu(tf.contrib.layers.batch_norm(conv3))

        conv4 = conv(bn3, W4, B4, name='conv4')
        bn4 = tf.nn.relu(tf.contrib.layers.batch_norm(conv4))

        conv5 = conv(bn4, W5, B5, name='conv5')
        bn5 = tf.nn.relu(tf.contrib.layers.batch_norm(conv5))

        conv6 = conv(bn5, W6, B6, name='conv6')
        bn6 = tf.nn.relu(tf.contrib.layers.batch_norm(conv6))

        conv7 = conv(bn6, W7, B7, name='conv7')
        bn7 = tf.nn.relu(tf.contrib.layers.batch_norm(conv7))

        conv8 = conv(bn7, W8, B8, name='conv8')
        bn8 = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.batch_norm(conv8)), keep_prob)

        pooled = max_pool_2x2(bn8)

        flat = tf.reshape(pooled,[batch_size, 12*12*T])
        output = tf.matmul(flat, W9) + B9

        # return tf.nn.softmax(output)
        return output
