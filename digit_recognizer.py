import os
#gpu_id = '1,2'
gpu_id = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
os.system('echo $CUDA_VISIBLE_DEVICES')

import tensorflow as tf
#config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.4


import random
import os.path
import numpy as np
import pandas as pd
import datetime
import tensorflow as tf

'''
the total number of training samples is 42000, and that of testing samples is 28000,
during training, 32000 for traning, and 10000 for evaluation
'''
total_size = 42000
batch_size = 200
train_total_size = 32000
test_batch_size = 400

def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 32],'weights')
    b_conv1 = bias_variable([32], 'biases')
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64], 'weights')
    b_conv2 = bias_variable([64], 'biases')
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024],'weights')
    b_fc1 = bias_variable([1024], 'biases')

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 10], 'weights')
    b_fc2 = bias_variable([10], 'biases')

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape,name):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)


def bias_variable(shape, name):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)

def convert2onehot(label,class_size):
  onehot = np.zeros((len(label),class_size))
  onehot[np.arange(len(label)), label] = 1
  return onehot

def main():
  f = open('train.csv') 
  df = pd.read_csv(filepath_or_buffer=f)
  index = [i for i in xrange(len(df))]
  random.shuffle(index)

  train_x = df.iloc[index[:train_total_size],1:].values
  train_y_ = df.iloc[index[:train_total_size],0].values
  train_y = convert2onehot(train_y_, 10)
  evaluate_x = df.iloc[index[train_total_size:],1:].values
  evaluate_y_ = df.iloc[index[train_total_size:],0].values
  evaluate_y = convert2onehot(evaluate_y_, 10)

  with tf.Graph().as_default():
    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Build the graph for the deep net
    y_conv, keep_prob = deepnn(x)

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    tf.summary.scalar('loss', cross_entropy)

    train_step = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    summary = tf.summary.merge_all()
    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
      sess.run(tf.global_variables_initializer())

      summary_writer = tf.summary.FileWriter('./fullylogs', sess.graph)

      iterator = int(train_total_size/batch_size)
      for j in range(400001):
        i = j%iterator
        if j % 2000 == 0:
          train_accuracy = accuracy.eval(feed_dict={x: evaluate_x ,y_: evaluate_y, keep_prob: 1.0})
          print('%s  step %d, training accuracy %g' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), j, train_accuracy))

          summary_str = sess.run(summary, feed_dict={x: evaluate_x ,y_: evaluate_y, keep_prob: 1.0})
          summary_writer.add_summary(summary_str, j)
          summary_writer.flush()

        if j % 10000 == 0:
          checkpoint_file = os.path.join('./fullylogs', 'model.ckpt')
          saver.save(sess, checkpoint_file, global_step=j)

        train_step.run(feed_dict={x: train_x[i*batch_size:(i+1)*batch_size],
            y_: train_y[i*batch_size:(i+1)*batch_size], keep_prob: 0.5})

#      for k in range((total_size-train_total_size)/test_batch_size):
#        train_accuracy += accuracy.eval(feed_dict={x: test_x[k*test_batch_size:(k+1)*test_batch_size],
#            y_: test_y[k*test_batch_size:(k+1)*test_batch_size], keep_prob: 1.0})
#      train_accuracy = train_accuracy/(total_size-train_total_size)*test_batch_size
#      print('%s  test accuracy %g' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), train_accuracy))

main()
