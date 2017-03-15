from __future__ import print_function
#import matplotlib.pyplot as plt
import numpy as np
# import cv2
# import csv
import os
import sys
import time
import struct
import h5py
import scipy.io as sio
# from scipy import ndimage
from numpy import linalg as LA
from IPython.display import display, Image
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import tensorflow as tf

import argparse


FLAGS = None
# Config the matplotlib backend as plotting inline in IPython
#%matplotlib inline
'''
def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  #train()
'''

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--max_steps', type=int, default=1000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
  parser.add_argument('--data_dir', type=str, default='',
                      help='Directory for storing input data')
  parser.add_argument('--log_dir', type=str, default='logs/',
                      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  #tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

print (tf.__version__)
# Load synthetic dataset
num_classes = 2

X = h5py.File('X_syn.mat')
y = h5py.File('y_syn.mat')

X = X['X']
X = np.transpose(X)
y = y['y_syn']
y = np.squeeze(np.asarray(y).astype(int))

# Randomize
np.random.seed(1337)  # for reproducibility
permutation = np.random.permutation(len(X))
#print(permutation)
X = [X[perm] for perm in permutation]
y = [y[perm] for perm in permutation]

#Select training and testing (75% and 25%)
X_train = X[:2250]
y_train = y[:2250]
X_test = X[2250:]
y_test = y[2250:]

sio.savemat('X_train.mat', {'X_train':X_train})
sio.savemat('y_train.mat', {'y_train':y_train})
sio.savemat('X_test.mat', {'X_test':X_test})
sio.savemat('y_test.mat', {'y_test':y_test})
#sio.savemat('c:/tmp/arrdata.mat', mdict={'arr': arr})
print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))

X_train = np.asarray(X_train).astype(float)
X_test = np.asarray(X_test).astype(float)
y_train = np.asarray(y_train).astype(float)
y_test = np.asarray(y_test).astype(float)


sio.savemat('y_train1.mat', {'y_train':y_train})
sio.savemat('y_test1.mat', {'y_test':y_test})

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

def extract_batch_size(_train, step, batch_size):
    # Function to fetch a "batch_size" amount of data from "(X|y)_train" data. 
    
    shape = list(_train.shape)
    #shape = list((batch_size, 1843200))
    shape[0] = batch_size
    #shape[1] = 1843200
    batch_s = np.empty(shape)

    for i in range(batch_size):
        # Loop index
        index = ((step-1)*batch_size + i) % len(_train)
        batch_s[i] = _train[index]
        #batch_s[i] = np.reshape(load_video(_train[index]), (1,1843200))

    return batch_s

def one_hot(y_):
    # Function to encode output labels from number indexes 
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    
    y_ = y_.reshape(len(y_))
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

import tensorflow as tf
import numpy as np

# Parameters
learning_rate = 0.0015
training_iters = 30000
batch_size = 300
display_step = 10

# Network Parameters
n_input = 200
n_classes = 2
dropout = 0.5 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, s, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    #return x
    #return tf.nn.relu(x)
    return (s**2)/(tf.square(x) + (s**2))


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')
# attach summaries to a tensor (for tensorboard visualization)
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)
        
# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 1, 200, 1])
#    tf.summary.image('input', image_shaped_input, 10) #add image summaries
    # Convolution Layer 1
#     conv1 = conv2d(x, weights['wc1'], biases['bc1'], strides=1)
#     # Max Pooling (down-sampling)
#     conv1 = maxpool2d(conv1, k=2)
    
    # Convolution Layer 1
    conv2 = conv2d(x, weights['wc2'], biases['bc2'], sigma['s1'], strides=1)
    # Max Pooling (down-sampling)
    #conv2 = maxpool2d(conv2, k=2)
    
    # Convolution Layer 1
#     conv3 = conv2d(x, weights['wc3'], biases['bc3'], strides=1)
#     # Max Pooling (down-sampling)
#     conv3 = maxpool2d(conv3, k=2)
    
#     concat_layers = tf.concat([conv1, conv2, conv3], axis=3)
#     print(concat_layers)
    #print(concat_layers)
    #Before fully-connected - square of all outputs
    #tensor_fro = tf.norm(conv2, ord='fro', axis=[1,2])
    #tensor_fro = tf.nn.relu(tensor_fro)
    #tensor_fro = (0.1**2)/(tf.square(tensor_fro) + (0.1**2))
    #print(tensor_fro)
    
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    with tf.name_scope('dropout'):
        tf.summary.scalar('dropout_keep_probability', dropout) #dropout keep_prob
        fc1 = tf.nn.dropout(fc1, dropout) #dropout keep_prob
        
#     fc2 = tf.reshape(conv2, [-1, weights['wd2'].get_shape().as_list()[0]])
#     fc2 = tf.add(tf.matmul(fc2, weights['wd2']), biases['bd2'])
#     fc2 = tf.nn.relu(fc2)
#     # Apply Dropout
#     fc2 = tf.nn.dropout(fc2, dropout)
    
#     fc3 = tf.reshape(conv3, [-1, weights['wd3'].get_shape().as_list()[0]])
#     fc3 = tf.add(tf.matmul(fc3, weights['wd3']), biases['bd3'])
#     fc3 = tf.nn.relu(fc3)
#     # Apply Dropout
#     fc3 = tf.nn.dropout(fc3, dropout)
    
    # Concatenation of the three FC layers
#     concat_layers = tf.concat([fc1, fc2, fc3], axis=1)
#     # Output, class prediction
#     print(fc1)
#     print(fc2)
#     print(fc3)
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    #out2 = tf.add(tf.matmul(fc2, weights['out2']), biases['out2'])
    #out3 = tf.add(tf.matmul(fc3, weights['out3']), biases['out3'])
    #out = (out1 + out2 + out3)/3
    return out

# Store layers weight & bias
#with tf.name_scope('weights'):
weights = {
	# 1x5 conv, 1 input, 96 maps
	#'wc1': tf.Variable(tf.random_normal([1, 3, 1, 32])),
	'wc2': tf.Variable(tf.random_normal([1, 3, 1, 32])),
    #'wc3': tf.Variable(tf.random_normal([1, 7, 1, 32])),
    'wd1': tf.Variable(tf.random_normal([1*200*32, 1024])),
    #'wd2': tf.Variable(tf.random_normal([100*100*32, 1024])),
    #'wd3': tf.Variable(tf.random_normal([100*100*32, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
    #'out2': tf.Variable(tf.random_normal([1024, n_classes])),
    #'out3': tf.Variable(tf.random_normal([1024, n_classes]))
    #'out': tf.Variable(tf.random_normal([96, n_classes]))
}
	#variables_
    
biases = {
    #'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([32])),
    #'bc3': tf.Variable(tf.random_normal([32])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    #'bd2': tf.Variable(tf.random_normal([1024])),
    #'bd3': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
    #'out2': tf.Variable(tf.random_normal([n_classes])),
    #'out3': tf.Variable(tf.random_normal([n_classes]))
}

sigma = {
    's1': tf.Variable(tf.random_normal([1]))
}

## add scalar and histogram summaries

        
# Add summary ops to collect data
#w_h = tf.image_summary("weights", weights['wc1'])
#b_h = tf.histogram_summary("biases", b)
# Construct model
#with tf.variable_scope('conv1') as scope_conv:
#pred = conv_net(x, weights, biases, keep_prob)
pred= conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
with tf.name_scope('cost'):
    with tf.name_scope('total'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
tf.summary.scalar('cost', cost)

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Siamese architecture
# Contrastive loss
# distance = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(out1,out2,out3)),1,keep_dims=True))
# distance = tf.div(self.distance, tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.out1),1,keep_dims=True)),tf.sqrt(tf.reduce_sum(tf.square(self.out2),1,keep_dims=True))))
# distance = tf.reshape(self.distance, [-1], name="distance")
# loss = self.contrastive_loss(self.input_y,self.distance, batch_size) 
# correct_predictions = tf.equal(self.distance, self.input_y)

# Evaluate model
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_pred'):
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar('accuracy', accuracy)

  # Merge all the summaries and write them out to /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)

# merged_summary_op = tf.merge_all_summaries()
# # Visualize conv1 features
# with tf.variable_scope('conv1') as scope_conv:
#   tf.get_variable_scope().reuse_variables()
#   #weights = tf.get_variable('weights')
#   #grid = put_kernels_on_grid (weights)
#   #tf.image_summary('conv1/features', grid, max_images=1)

# Initializing the variables
#init = tf.global_variables_initializer()

init=tf.initialize_all_variables() #tf.global_variables_initializer()


# Launch the graph
training_iters = 30000
display_step = 10
#y_train_encoded = one_hot(y_train)
with tf.Session() as sess:
    sess.run(init)
    #summary_writer = tf.train.SummaryWriter('/Users/angelsrates/Documents/keras_try/', graph_def=sess.graph_def)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size <= training_iters:
        batch_x = extract_batch_size(X_train,step,batch_size)
        batch_y = one_hot(extract_batch_size(y_train,step,batch_size))
        k = FLAGS.dropout
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout}) #dropout k
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")
    
    # Calculate accuracy for 128 mnist test images
    test_label = one_hot(y_test)
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: X_test, y: test_label, keep_prob: 1.}))

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
