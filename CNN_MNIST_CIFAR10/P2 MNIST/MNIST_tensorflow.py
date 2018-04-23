import numpy as np
import h5py
import time

import tensorflow as tf
#sess = tf.InteractiveSession()

#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

#this trains a single layer neural network for MNIST


#load MNIST data..reformat data into TensorFlow's required format
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
L_Y_train = len(y_train)
y_train2 = np.int32( np.zeros( (L_Y_train, 10 ) ) )
for i in range(L_Y_train):
    y_train2[i,y_train[i]] = 1
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0]  ) )
L_Y_test = len(y_test)
y_test2 = np.int32( np.zeros( (L_Y_test, 10 ) ) )
for i in range(L_Y_test):
    y_test2[i,y_test[i]] = 1

MNIST_data.close()

#number of hidden units
H = 250
#number of epochs
num_epochs = 200
batch_size = 1000
#learning rate
LR = .1

#model is trained using stochastic gradient descent

#available devices: gpu:0 (default), cpu:0, cpu:1, ..., cpu:15
#by default, model is trained on gpu:0 (the best available device)


time1 = time.time()
# with tf.device('/gpu:0'):
#with tf.device('/cpu:0'):
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#single layer neural network
W1 = tf.Variable(tf.random_normal([784,H], stddev=1.0/np.sqrt(784) ) )
b1 = tf.Variable( tf.zeros([H] ) )
W2 = tf.Variable( tf.random_normal([H, 10], stddev=1.0/np.sqrt(H) )  ) 
b2 = tf.Variable( tf.zeros([10]) )
                           
h1 = tf.nn.relu( tf.matmul(x, W1) + b1 )
y = tf.nn.softmax(tf.matmul(h1,W2) + b2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.GradientDescentOptimizer(LR).minimize(cross_entropy)

sess = tf.Session()

sess.run(tf.global_variables_initializer())     
for epochs in range(num_epochs):
    for i in range(0, L_Y_train, batch_size):
        #train_step.run(feed_dict={x: x_train[i:i+batch_size,:], y_: y_train2[i:i+batch_size,:]  })
        _, train_accuracy, loss = sess.run([train_step, accuracy, cross_entropy],feed_dict={x: x_train[i:i+batch_size,:], y_: y_train2[i:i+batch_size,:]})
        # _ = sess.run([train_step],feed_dict={x: x_train[i:i+batch_size,:], y_: y_train2[i:i+batch_size,:]})
        

    print("epoch:%d accuracy:%f,loss:%f" % (epochs,train_accuracy,loss))

test_accuracy = sess.run([accuracy],feed_dict={x: x_test, y_: y_test2})
print("Test Accuracy: %f" % test_accuracy[0])



time2 = time.time()
total_time = time2 - time1
print("Time: %f" % total_time)
