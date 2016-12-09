from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import numpy as np
import sys
from pprint import pprint
# Import data
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
FLAGS = None

def reduce_train_data(data):
  '''only keeps the first 100 examples of every class in the train set'''
  train = data.train
  labels = train._labels
  pprint(labels)
  images = train._images
  count_arr = np.zeros((10))
  images_reduced = []
  labels_reduced = []
  for i in range(0,labels.shape[0]):
    l = np.nonzero(labels[i])
    if count_arr[l] < 100:
      count_arr[l] = count_arr[l] + 1
      images_reduced.append(images[i])
      labels_reduced.append(labels[i])

  images_reduced = np.asarray(images_reduced)
  labels_reduced = np.asarray(labels_reduced)
  data.train._images = images_reduced
  data.train._labels = labels_reduced
  return data

def main(_):
  '''
  INSTRUCTIONS:
  For a standard nnet, set hyper_mode to False.
  The comments beginning with #hyp_1 and #hyp_2 give two simple examples of hypernetworks.
  Feel free to experiment with multilayered hypernetworks, different types of activations, etc.
  '''
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  mnist = reduce_train_data(mnist)
  
  num_output = 10
  num_input = 784
  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  #hyper_mode = False
  hyper_mode = True
  if hyper_mode:
    #hyp_1
    w_hyp = tf.get_variable('w_hyp',[1024,1],initializer=tf.truncated_normal_initializer(stddev=0.01))
    w2_hyp = tf.get_variable('w_hyp2',[1,1568],initializer=tf.truncated_normal_initializer(stddev=0.01))
    b_hyp = tf.get_variable('b2ab',[1568], initializer=tf.truncated_normal_initializer(stddev=0.01)) 
    w = tf.matmul(w_hyp, w2_hyp) + b_hyp
    W = tf.reshape(w, [784, 2048])
    #hyp_2
    #w_hyp = tf.get_variable('w_hyp',[784,1],initializer=tf.truncated_normal_initializer(stddev=0.01))
    #w2_hyp = tf.get_variable('w_hyp2',[1,2048],initializer=tf.truncated_normal_initializer(stddev=0.01))
    #b_hyp = tf.get_variable('b2ab',[2048], initializer=tf.constant_initializer(0.0)) 
    #W = tf.matmul(w_hyp, w2_hyp) + b_hyp
    
  else:
    W = tf.get_variable('W',[784,2048],initializer=tf.truncated_normal_initializer(stddev=0.01))
  b = tf.Variable(tf.zeros([2048]))
  y1 = tf.sigmoid(tf.matmul(x, W) + b)

  W2 = tf.Variable(tf.zeros([2048, 10]))
  b2 = tf.Variable(tf.zeros([10]))
  y = tf.matmul(y1,W2) + b2

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
  sess = tf.InteractiveSession()
  # Train
  tf.initialize_all_variables().run()
  for _ in range(10):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))
  t_vars = tf.trainable_variables()
  count_t_vars = 0
  for var in t_vars:
    var = tf.Print(var, [var], message="This is a: ")
    num_param = np.prod(var.get_shape().as_list())
    count_t_vars = count_t_vars + num_param
    #print(var.name +',' + var.get_shape()+ ','+ num_param)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/data',
                      help='Directory for storing data')
  FLAGS = parser.parse_args()
  tf.app.run()
