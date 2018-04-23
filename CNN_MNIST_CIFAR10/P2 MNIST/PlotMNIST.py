import numpy as np

import time

#load MNIST data
import h5py
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0]  ) )

MNIST_data.close()

###########################################################################
#####plot the images..

def PlotImage(vector_in, time_to_wait):
    array0 = 255.0*np.reshape(vector_in, (28,28) )
    from matplotlib import pyplot as plt
    plt.ion()
    plt.show(block=False)
    plt.figure(figsize= (2,2) )
    plt.imshow(array0, cmap='Greys_r')
    plt.draw()
    plt.show()
    time.sleep(time_to_wait)
    plt.close('all')
    
    
#time (in seconds) between each image showing up on the screen
time_to_wait = 1
#number of images from the dataset that will be plotted
N = 100
#plot the images
for i in range(0, N):
    PlotImage(x_train[i], time_to_wait)