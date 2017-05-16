
# coding: utf-8

# In[1]:

from __future__ import print_function
import matplotlib.pyplot as plt
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

# Config the matplotlib backend as plotting inline in IPython
get_ipython().magic(u'matplotlib inline')


# In[3]:

import scipy.io
# Load synthetic dataset
num_classes = 3

# 60 samples
#X = h5py.File('/Users/angelsrates/Documents/PhD/Robust Systems Lab/Activity Recognition/Code/poles_data.mat')
#y = h5py.File('/Users/angelsrates/Documents/PhD/Robust Systems Lab/Activity Recognition/Code/poles_y.mat')

# 300 samples
X = scipy.io.loadmat('/Users/angelsrates/Documents/activity-recog-synthetic/poles_data2.mat')
y = scipy.io.loadmat('/Users/angelsrates/Documents/activity-recog-synthetic/poles_y2.mat')

X = X['data']
X = np.squeeze(np.transpose(X))
y = y['label']
y = y - 1
y = np.squeeze(y)


# In[4]:

np.random.seed(4294967295)
permutation = np.random.permutation(len(X))
X = [X[perm] for perm in permutation]
y = [y[perm] for perm in permutation]


# In[5]:

#Select training and testing (75% and 25%)
y = [int(i) for i in y]
X_train = np.asarray(X[:225])
y_train = np.asarray(y[:225])
X_test = np.asarray(X[225:])
y_test = np.asarray(y[225:])

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[192]:

sys_par = np.array([[-1, 0.823676337910219, -1], [-1,-1.93592782488463,-1]])


# In[6]:

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


# In[24]:

from sklearn import linear_model

#alpha = 0.1
def np_sparseLoss(y,p,alpha):
    #Assume p is real
    N = y.shape[0]
    k = p.shape[1]
    print(p)
    W = np.zeros((N,k))
    pw_idx = np.arange(1, N+1, 1)
    #print(pw_idx.shape)
    # Define vocabulary on set of poles
    for i in range(k):
        W[:,i] = np.power(np.squeeze(np.full((1, N), np.squeeze(p[0,i]))), pw_idx)
    # ADMM - Lasso
    clf = linear_model.Lasso(alpha=alpha)
    clf.fit(W, y)
    linear_model.Lasso(alpha=alpha, copy_X=True, fit_intercept=True, max_iter=1000,
       normalize=True, positive=False, precompute=False, random_state=None,
       selection='cyclic', tol=0.0001, warm_start=False)
    coeff = clf.coef_
    coeff = np.reshape(coeff, [k,1])
    print(coeff)
    return coeff

#s = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
#c = np_sparseLoss(s,p,alpha)


# In[42]:

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

def coeff_grad(y,p,c, grad):
    #y = op.inputs[0] 
    #p = op.inputs[1]
    #c = op.outputs[0]
    #W_shape = W.get_shape().as_list()
    y_shape = y.get_shape().as_list()
    p_shape = p.get_shape().as_list()
    N = y_shape[0]
    K = p_shape[1]
    # W Calculation
    impulse = []
    idx = tf.cast(tf.stack(np.arange(1, N+1, 1)), tf.float64)
    for cc in range(K):
        impulse.append(tf.pow(tf.tile(p[:,cc], [50], name=None), idx , name=None))
    W = tf.cast(tf.reshape(tf.stack(impulse, axis=1), (N,K)), tf.float64)
    WW = tf.matrix_inverse(tf.matmul(tf.transpose(W), W))
    Wty = tf.matmul(tf.transpose(W), y)
    WWc = tf.matmul(WW, c)
    output_dW = []
    # Grad wrt W
    for i in range(N):
        for j in range(K):
            output_dWty = []
            output_dWWc = []
            for n in range(K):
                gr1 = tf.gradients(Wty[n,:], [W[i,j]])
                gr1 = [tf.constant(0, dtype=tf.float64) if t == None else t for t in gr1]
                gr2 = tf.gradients(WWc[n,:], [W[i,j]])
                gr2 = [tf.constant(0, dtype=tf.float64) if t == None else t for t in gr2]
                output_dWty.append(gr1)
                output_dWWc.append(gr2)
            gr = tf.matmul(WW, tf.subtract(tf.stack(output_dWty), tf.stack(output_dWWc)))
            output_dW.append(gr)
    dW = tf.reshape(tf.squeeze(tf.stack(output_dW)), [N, K, K])
    
    # Grad wrt p
    grp = []
    for k in range(K):
        output_dp = []
        for i in range(N):
            output_dp.append(tf.multiply(tf.reshape(tf.multiply(tf.cast(i, tf.float64), tf.pow(p[0, k],tf.cast(i-1, tf.float64))), [1]), tf.reshape(dW[i,k,:], [K,1])))
        grp.append(tf.add_n(output_dp))
    dp = tf.stack(grp)
    dp_list = []
    for j in range(K):
        dp_list.append(tf.reduce_sum(tf.multiply(dp[j,:,:], grad)))
    dp = tf.reshape(tf.stack(dp_list), [1, K])
    print('dc/dp size:', dp.get_shape())
    
    #dW = tf.reshape(dW, [N*K,K,1])
    #dW_list = []
    #for j in range(N*K):
    #    dW_list.append(tf.reduce_sum(tf.multiply(dW[j,:,:], grad)))
    #dW = tf.reshape(tf.stack(dW_list), [N, K])
    #print('dc/dW size:', dW.get_shape())
    
    # Grad wrt y
    dy = tf.matmul(WW, tf.transpose(W))
    dy_list = []
    for j in range(N):
        dy_list.append(tf.reduce_sum(tf.multiply(dy[:,j], grad)))
    dy = tf.reshape(tf.stack(dy_list), [N, 1])
    print('dc/dy size:', dy.get_shape())
    
    # Grad wrt alpha   
    dalpha = tf.matmul(tf.scalar_mul(tf.constant(-1, dtype=tf.float64), WW), tf.sign(c))
    dalpha = tf.reduce_sum(tf.multiply(dalpha, grad))
    print('dc/dalpha size:', dalpha.get_shape())
    
    return dy, dp, dalpha


# In[25]:

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):

    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1000000000000000))

    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


# In[30]:

from tensorflow.python.framework import ops

def tf_coeff_grad(y,p,alpha, name=None):

    with ops.op_scope([y,p,alpha], name, "CoeffGrad") as name:
        z = py_func(np_sparseLoss,
                        [y,p,alpha],
                        [tf.double],
                        name=name,
                        grad=coeff_grad)  # <-- here's the call to the gradient
        return z[0]


# In[46]:

from scipy import signal
#import control
from scipy.signal import step2
import math

# Parameters
learning_rate = 0.0015
#training_iters = 45000
batch_size = 1
#display_step = 10

# Network Parameters
n_input = 50
n_classes = 3
N = 50
#dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float64, [n_input, 1])
y = tf.placeholder(tf.float64, [1, n_classes])
grad = tf.constant(0, dtype=tf.float64)
#y = tf.placeholder(tf.int32, [1,1])

def index_along_every_row(array, index):
    N,_ = array.shape 
    return array[np.arange(N), index]

def build_hankel_tensor(x, nr, nc, N, dim):
    cidx = np.arange(0, nc, 1)
    ridx = np.transpose(np.arange(1, nr+1, 1))
    Hidx = np.transpose(np.tile(ridx, (nc,1))) + dim*np.tile(cidx, (nr,1))
    Hidx = Hidx - 1
    arr = tf.reshape(x[:], (1,N))
    return tf.py_func(index_along_every_row, [arr, Hidx], [tf.float64])[0]

def build_hankel(x, nr, nc, N, dim):
    cidx = np.arange(0, nc, 1)
    ridx = np.transpose(np.arange(1, nr+1, 1))
    Hidx = np.transpose(np.tile(ridx, (nc,1))) + dim*np.tile(cidx, (nr,1))
    Hidx = Hidx - 1
    arr = x[:]
    return arr[Hidx]

# Create model
def poles_net(x,grad):
    # Reshape input picture
    #x = tf.reshape(x, shape=[-1, , 50, 1])
    # Change accordingly
    dim = 1
    N = 50
    num_poles = 2
    # Complex poles
    #idx = tf.cast(tf.stack(np.arange(1, N+1, 1)), tf.complex128)
    #p11 = tf.multiply(tf.cast(tf.sqrt(weights['r11']), tf.complex128), tf.exp(tf.complex(tf.constant(0, tf.float64), weights['theta11'])))
    #p12 = tf.multiply(tf.cast(tf.sqrt(weights['r12']), tf.complex128), tf.exp(tf.complex(tf.constant(0, tf.float64), -weights['theta12'])))
    #p21 = tf.multiply(tf.cast(tf.sqrt(weights['r21']), tf.complex128), tf.exp(tf.complex(tf.constant(0, tf.float64), weights['theta21'])))
    #p22 = tf.multiply(tf.cast(tf.sqrt(weights['r22']), tf.complex128), tf.exp(tf.complex(tf.constant(0, tf.float64), -weights['theta22'])))   
    #y11 = tf.pow(tf.tile(p11, [50], name=None), idx , name=None)
    #y12 = tf.pow(tf.tile(p12, [50], name=None), idx, name=None)
    #y21 = tf.pow(tf.tile(p21, [50], name=None), idx, name=None)
    #y22 = tf.pow(tf.tile(p22, [50], name=None), idx, name=None)
    #W = tf.cast(tf.reshape(tf.stack([y11, y21, y12, y22], 1), (N,4)), tf.float64)
    
    # Real poles
    idx = tf.cast(tf.stack(np.arange(1, N+1, 1)), tf.float64)
    p1 = weights['real1']
    p2 = weights['real2']
    p3 = weights['real3']
    p = tf.stack([p1, p2, p3], 1)
    
    #alpha = tf.constant([0.1])
    alpha = weights['alpha']
    #a = tf.matmul(W, W, adjoint_a=True)
    #c = tf.matrix_inverse(tf.cast(a, tf.float64), adjoint=False, name=None)
    #coeff = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(W), W)), tf.transpose(W)), tf.reshape(x, (N,1)))
    #alpha_ind = tf.reshape(tf.stack([alpha_ind1, alpha_ind2]), (1,2))
    #coeff = tf.matrix_solve_ls(W, tf.reshape(x, (N,1)), fast=False, l2_regularizer=0.002, name=None)
    #x = tf.cast(tf.reshape(x, (N,1)), tf.complex128)
    #coeff = tf.transpose(coeff)
    coeff = tf_coeff_grad(x,p,alpha)
    print(coeff)
    out = tf.add(tf.matmul(tf.transpose(tf.cast(coeff, tf.float32)), weights['out']), biases['out'])
    diff = tf.subtract(tf.cast(out, tf.float64), y)
    grad = coeff_grad(x,p,coeff,grad)
    return [coeff, out]

weights = {
    'r11': tf.Variable(tf.random_uniform([1], minval=(0.02)**2, maxval=(1)**2, dtype=tf.float64)), # Complex poles
    'r12': tf.Variable(tf.random_uniform([1], minval=(0.02)**2, maxval=(1)**2, dtype=tf.float64)),
    'theta11': tf.Variable(tf.random_uniform([1], minval=0, maxval=math.pi, dtype=tf.float64)),
    'theta12': tf.Variable(tf.random_uniform([1], minval=0, maxval=math.pi, dtype=tf.float64)),
    'r21': tf.Variable(tf.random_uniform([1], minval=(0.02)**2, maxval=(1)**2, dtype=tf.float64)),
    'r22': tf.Variable(tf.random_uniform([1], minval=(0.02)**2, maxval=(1)**2, dtype=tf.float64)),
    'theta21': tf.Variable(tf.random_uniform([1], minval=0, maxval=math.pi, dtype=tf.float64)),
    'theta22': tf.Variable(tf.random_uniform([1], minval=0, maxval=math.pi, dtype=tf.float64)),
    'real1': tf.Variable(tf.random_uniform([1], minval=-1, maxval=1, dtype=tf.float64)), # Real poles
    'real2': tf.Variable(tf.random_uniform([1], minval=-1, maxval=1, dtype=tf.float64)),
    'real3': tf.Variable(tf.random_uniform([1], minval=-1, maxval=1, dtype=tf.float64)),
    'alpha' : tf.Variable(tf.constant(0.1, dtype=tf.float64)),
    #'sys_par1': tf.Variable(tf.random_normal([1], dtype=tf.float64)),
    #'sys_par2': tf.Variable(tf.random_normal([1], dtype=tf.float64)),
    'out': tf.Variable(tf.random_normal([2, n_classes]))
}
    
biases = {
    'out': tf.Variable(tf.random_normal([1, n_classes]))
}

[coeff, pred]= poles_net(x,grad)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
#cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=pred, onehot_labels=y))
#cost = tf.reduce_mean(tf.losses.mean_squared_error(predictions=pred, labels=y))
#cost = tf.subtract(pred, labels)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()


# In[14]:

y_test = one_hot(y_test)


# In[32]:

# Launch the graph
training_iters = X_train.shape[0]*10
display_step = 1
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    train_acc = 0
    while step * batch_size <= training_iters:
        batch_x = np.reshape(extract_batch_size(X_train,step,batch_size), [50, 1])
        batch_y = extract_batch_size(one_hot(y_train),step,batch_size)
        #batch_y = np.reshape(extract_batch_size(y_train,step,batch_size), (1,1))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y})
            train_acc += acc 
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " +                   "{:.6f}".format(loss) + ", Training Accuracy= " +                   "{:.5f}".format(acc))
        step += 1
    print('Final Training Accuracy:', train_acc/(X_train.shape[0]*10))
    print("Optimization Finished!")
    
    acc = 0
    for i in range(X_test.shape[0]):
        test = np.reshape(X_test[i,:], [50,1])
        print(test.shape)
        label = np.reshape(y_test[i,:], (1,3))
        #label = np.reshape(y_test[i], (1,1))
        print("Trajectory:", i,             sess.run([coeff], feed_dict={x: test, y: label}))
        print("Testing Accuracy:",             sess.run(accuracy, feed_dict={x: test, y: label}))
        acc += sess.run(accuracy, feed_dict={x: test, y: label})
    print('Final Testing Accuracy:', acc/X_test.shape[0])


# In[ ]:

# Confusion matrix code

pred = multilayer_perceptron(x, weights, biases)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    for epoch in xrange(150):
            for i in xrange(total_batch):
                    train_step.run(feed_dict = {x: train_arrays, y: train_labels})
                    avg_cost += sess.run(cost, feed_dict={x: train_arrays, y: train_labels})/total_batch         
            if epoch % display_step == 0:
                    print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

    #metrics
    y_p = tf.argmax(pred, 1)
    val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x:test_arrays, y:test_label})

    print "validation accuracy:", val_accuracy
    y_true = np.argmax(test_label,1)
    print "Precision", sk.metrics.precision_score(y_true, y_pred)
    print "Recall", sk.metrics.recall_score(y_true, y_pred)
    print "f1_score", sk.metrics.f1_score(y_true, y_pred)
    print "confusion_matrix"
    print sk.metrics.confusion_matrix(y_true, y_pred)
    fpr, tpr, tresholds = sk.metrics.roc_curve(y_true, y_pred)

