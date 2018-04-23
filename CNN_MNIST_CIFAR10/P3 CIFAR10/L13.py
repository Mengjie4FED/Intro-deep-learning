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


def cifar10_conv_L13(X, keep_prob, reuse=False):
    with tf.variable_scope('cifar10_conv_L13'):
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

        E = 400
        F = 400
        G = 400
        H = 400
        I = 400

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

        W9 = tf.get_variable('D_W9', [5, 5, T, E], initializer=tf.contrib.layers.xavier_initializer())
        B9 = tf.get_variable('D_B9', [E], initializer=tf.constant_initializer())

        W10 = tf.get_variable('D_W10', [5, 5, E, F], initializer=tf.contrib.layers.xavier_initializer())
        B10 = tf.get_variable('D_B10', [F], initializer=tf.constant_initializer())

        W11 = tf.get_variable('D_W11', [5, 5, F, G], initializer=tf.contrib.layers.xavier_initializer())
        B11 = tf.get_variable('D_B11', [G], initializer=tf.constant_initializer())

        W12 = tf.get_variable('D_W12', [5, 5, G, H], initializer=tf.contrib.layers.xavier_initializer())
        B12 = tf.get_variable('D_B12', [H], initializer=tf.constant_initializer())

        W13 = tf.get_variable('D_W13', [5, 5, H, I], initializer=tf.contrib.layers.xavier_initializer())
        B13 = tf.get_variable('D_B13', [I], initializer=tf.constant_initializer())

        W14 = tf.get_variable('D_W14', [12*12*I, 10], initializer=tf.contrib.layers.xavier_initializer())
        B14 = tf.get_variable('D_B14', [10], initializer=tf.constant_initializer())

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
        bn8 = tf.nn.relu(tf.contrib.layers.batch_norm(conv8))

        conv9 = conv(bn8, W9, B9, name='conv4')
        bn9 = tf.nn.relu(tf.contrib.layers.batch_norm(conv9))

        conv10 = conv(bn9, W10, B10, name='conv10')
        bn10 = tf.nn.relu(tf.contrib.layers.batch_norm(conv10))

        conv11 = conv(bn10, W11, B11, name='conv11')
        bn11 = tf.nn.relu(tf.contrib.layers.batch_norm(conv11))

        conv12 = conv(bn11, W12, B12, name='conv12')
        bn12 = tf.nn.relu(tf.contrib.layers.batch_norm(conv12))

        conv13 = conv(bn12, W13, B13, name='conv13')
        bn13 = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.batch_norm(conv13)), keep_prob)

        pooled = max_pool_2x2(bn13)

        flat = tf.reshape(pooled,[batch_size, 12*12*I])
        output = tf.matmul(flat, W14) + B14

        # return tf.nn.softmax(output)
        return output
