from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
import gzip
import numpy
import json
import os
import tensorflow as tf
import pickle
import numpy as np
from sklearn.cross_validation import train_test_split
import cv2
import pandas as pd
import sys

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_integer('n_bundle', 1, 'Number of bundles to upload.')
flags.DEFINE_integer('validation_size', 20000, 'Number of bundles to upload.')
flags.DEFINE_string('data_dir', '~/tmp/data', 'Directory for storing data')
flags.DEFINE_string('summaries_dir', '~/tmp/mnist_logs', 'Summaries directory')

# flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
#                                          'for unit testing.')
# flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
# flags.DEFINE_float('dropout', 0.9, 'Keep probability for training dropout.')

validation_size = FLAGS.validation_size


def train():
    class DataSet(object):

        def __init__(self,
                     images,
                     labels,
                     fake_data=False,
                     one_hot=False,
                     dtype=dtypes.float32,
                     reshape=True):
            """Construct a DataSet.
            one_hot arg is used only if fake_data is true.  `dtype` can be either
            `uint8` to leave the input as `[0, 255]`, ocdr `float32` to rescale into
            `[0, 1]`.
            """
            dtype = dtypes.as_dtype(dtype).base_dtype
            if dtype not in (dtypes.uint8, dtypes.float32):
                raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                                dtype)
            if fake_data:
                self._num_examples = 10000
                self.one_hot = one_hot
            else:
                assert images.shape[0] == labels.shape[0], (
                    'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
                self._num_examples = images.shape[0]
                images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])

                if dtype == dtypes.float32:
                    # Convert from [0, 255] -> [0.0, 1.0].
                    images = images.astype(numpy.float32)
                    images = numpy.multiply(images, 1.0 / 255.0)
            self._images = images
            self._labels = labels
            self._epochs_completed = 0
            self._index_in_epoch = 0

        @property
        def images(self):
            return self._images

        @property
        def labels(self):
            return self._labels

        @property
        def num_examples(self):
            return self._num_examples

        @property
        def epochs_completed(self):
            return self._epochs_completed

        def next_batch(self, batch_size, fake_data=False):
            """Return the next `batch_size` examples from this data set."""
            if fake_data:
                fake_image = [1] * 784
                if self.one_hot:
                    fake_label = [1] + [0] * 9
                else:
                    fake_label = 0
                return [fake_image for _ in xrange(batch_size)], [
                    fake_label for _ in xrange(batch_size)
                    ]
            start = self._index_in_epoch
            self._index_in_epoch += batch_size
            if self._index_in_epoch > self._num_examples:
                # Finished epoch
                self._epochs_completed += 1
                # Shuffle the data
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._images = self._images[perm]
                self._labels = self._labels[perm]
                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size
                assert batch_size <= self._num_examples
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]

    def read_data_sets(data_dict, fake_data=False, one_hot=False, dtype=dtypes.float32, reshape=True,
                       validation_size=5000):
        train_images = np.array(data_dict['X_train'])
        train_labels = np.array(data_dict['y_train'])
        test_images = np.array(data_dict['X_test'])
        test_labels = np.array(data_dict['y_test'])
        if not 0 <= validation_size <= len(train_images):
            raise ValueError(
                'Validation size should be between 0 and {}. Received: {}.'.format(len(train_images), validation_size))

        validation_images = train_images[:validation_size]
        validation_labels = train_labels[:validation_size]
        train_images = train_images[validation_size:]
        train_labels = train_labels[validation_size:]

        train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
        validation = DataSet(validation_images, validation_labels, dtype=dtype, reshape=reshape)
        test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape)
        return base.Datasets(train=train, validation=validation, test=test)

    def flatten_matrix(matrix):
        vector = matrix.flatten(1)
        vector = vector.reshape(1, len(vector))
        return vector

    def split_data(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        return {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}

    def normalizer(label):
        return ((label - np.min(label)) / (np.max(label) - np.min(label))) * 0.8 + 0.1

    def read_data_from_processed_pickle(pickle_data):
        print("read processed pickle...")
        with open("../processed_pickle/%s" % pickle_data, 'rb') as handle:
            data = pickle.load(handle,encoding='latin1')
            # data = pickle.load(handle)
            return data

    path = os.getcwd() + "/../processed_pickle"
    print("path: %s" % path)
    processed_pickles = [item for item in os.listdir(path) if item.endswith(".pickle")]
    processed_pickles = processed_pickles[:FLAGS.n_bundle]

    processed_data = [[], []]
    for item in processed_pickles:
        print("getting %s" % item)
        tmp_data = read_data_from_processed_pickle(item)
        processed_data[0].extend(tmp_data[0])
        processed_data[1].extend(tmp_data[1])

    # WIDTH = 280
    # HEIGHT = 212
    WIDTH = 104
    HEIGHT = 144
    channel = 1
    n_class = 14

    images_data = []
    labels_data = []

    for idx in range(len(processed_data[0])):
        image = processed_data[0][idx]['image']
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, (HEIGHT, WIDTH))
        # img_gray = np.vstack((img_gray, np.zeros((2, WIDTH))))
        images_data.append(img_gray)
        labels_data.append(processed_data[1][idx]['label'])



    labels_data_norm = pd.DataFrame(labels_data).apply(normalizer, 0).as_matrix()

    data_dict = split_data(images_data, labels_data_norm)

    sample_size = len(data_dict['y_train'])

    torcs_data = read_data_sets(data_dict, one_hot=False, validation_size=validation_size)

    # Weight Initialization
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # Convolution and Pooling
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, [None, WIDTH * HEIGHT * channel])
    W = tf.Variable(tf.zeros([WIDTH * HEIGHT * channel, n_class]))
    b = tf.Variable(tf.zeros([n_class]))
    y_ = tf.placeholder(tf.float32, [None, n_class])

    sess.run(tf.initialize_all_variables())
    # sess = tf.InteractiveSession()
    # init = tf.initialize_all_variables()
    # sess = tf.Session()
    # sess.run(init)
    # init = tf.initialize_all_variables()
    # sess = tf.Session()
    # sess.run(init)

    x_image = tf.reshape(x, [-1, WIDTH, HEIGHT, channel])

    with tf.name_scope('input_reshape'):
        tf.image_summary('input', x_image, 10)


    ## first
    W_conv1 = weight_variable([5, 5, channel, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    ## second
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    ## third
    W_conv3 = weight_variable([5, 5, 64, 128])
    b_conv3 = bias_variable([128])

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)

    h_pool2 = max_pool_2x2(h_conv3)

    ## forth

    W_conv4 = weight_variable([5, 5, 128, 256])
    b_conv4 = bias_variable([256])

    h_conv4 = tf.nn.relu(conv2d(h_pool2, W_conv4) + b_conv4)

    h_pool3 = max_pool_2x2(h_conv4)

    W_fc1 = weight_variable([(HEIGHT // 8) * (WIDTH // 8) * 256, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool3, [-1, (HEIGHT // 8) * (WIDTH // 8) * 256])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # keep_prob = tf.placeholder(tf.float32)
    # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, n_class])
    b_fc2 = bias_variable([n_class])
    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

    sample_size = sample_size - validation_size

    with tf.name_scope('mse'):
        cost_function = tf.reduce_sum(tf.pow(y_ - y_conv, 2)) / (2 * sample_size)  # L2 loss
    tf.scalar_summary('mse', cost_function)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cost_function)
    mse = tf.reduce_sum(tf.pow(y_ - y_conv, 2)) / (2 * sample_size)

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train',
                                          sess.graph)
    test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')

    # tf.initialize_all_variables().run()

###    with tf.name_scope('input_reshape'):
        #tf.image_summary('input', images_data, 10)


    def feed_dict():
        batch = torcs_data.train.next_batch(100)
        return {x: batch[0], y_: batch[1]}


    # accuracy_lst = []
    sess.run(tf.initialize_all_variables())
    for i in range(FLAGS.max_steps):

        if i % 10 == 0:
            # train_accuracy = mse.eval(session=sess, feed_dict=feed_dict)
            # with open('result.txt', 'w') as outfile:
            #     json.dump(str(accuracy_lst), outfile)
            # print("step %d, training mse %g" % (i, train_accuracy))
            # accuracy_lst.append(train_accuracy)
            summary, acc = sess.run([merged, cost_function], feed_dict=feed_dict())
            test_writer.add_summary(summary, i)
            print('MSE at step %s: %s' % (i, acc))
        else:  # Record train set summaries, and train
            if i % 100 == 99:  # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      feed_dict=feed_dict(),
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run metadata for', i)
            else:  # Record a summary
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict())
                train_writer.add_summary(summary, i)
        # train_step.run(session=sess, feed_dict={x: batch[0], y_: batch[1]})

    train_writer.close()
    test_writer.close()

    # with open('result.txt', 'w') as outfile:
    #     json.dump(str(accuracy_lst), outfile)

    # accuracy = mse.eval(session=sess, feed_dict={x: torcs_data.test.images, y_: torcs_data.test.labels})
    # print(accuracy)

    # with open('result_test.txt', 'w') as outfile2:
    #     json.dump(str(accuracy), outfile2)
    #
    # saver = tf.train.Saver()
    # saver.save(sess, 'model')


def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)

    print("data_dir: %s" % FLAGS.data_dir)
    print("n_bundle: %s" % FLAGS.n_bundle)
    print("max_steps: %s" % FLAGS.max_steps)

    train()


if __name__ == '__main__':
    tf.app.run()
