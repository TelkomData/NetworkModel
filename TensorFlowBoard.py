from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
from log_reg_exc import load_exchange

FLAGS = None

def train(X_train, X_test, y_train, y_test, save_files_to):

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

    def feed_dict(i):
        if isinstance(i, int):
            xs, ys = X_train[(i*n_step):(i+1)*n_step], y_train[(i*n_step):(i+1)*n_step]
        else:
            xs, ys = X_test, y_test
        return {X: xs, Y: ys}

    n_step = FLAGS.n_step

    sess = tf.InteractiveSession()

    num_features = X_train.shape[1]
    if len(y_train.shape) == 1:
        num_classes = 1
    else:
        num_classes = y_train.shape[1]

    X = tf.placeholder(tf.float32, [None, num_features])
    Y = tf.placeholder(tf.float32, [None, num_classes])

    with tf.name_scope('W'):
        W = tf.Variable(tf.zeros([num_features, num_classes]))
        variable_summaries(W)    
        
    with tf.name_scope('b'):
        b = tf.Variable(tf.zeros([num_classes]))
        variable_summaries(b)
        
    with tf.name_scope('y_hat'):
        y_hat = tf.nn.softmax(tf.matmul(X, W) + b)
        variable_summaries(y_hat)    

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=Y))
    train_step = tf.train.GradientDescentOptimizer(0.4).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(y_hat, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # create the summary logs
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(save_files_to + 'train', sess.graph)
    test_writer = tf.summary.FileWriter(save_files_to + 'test')

    # save the model

    
    sess.run(tf.global_variables_initializer())

    for i in range(FLAGS.max_steps):   
        
        train, summary = sess.run([train_step, merged], feed_dict=feed_dict(i))
        train_writer.add_summary(summary, i)
        acc, summary = sess.run([accuracy, merged], feed_dict=feed_dict(False))
        test_writer.add_summary(summary, i)
        print('Accuracy at step {:d}: {:0.2f}%'.format(i, acc*100))

        if (i * n_step) > len(X_train):
            break
    return sess.run(W)

def main(_):
    save_files_to = FLAGS.data_dir + 'models/' + FLAGS.exchange +'/'
    if tf.gfile.Exists(save_files_to):
        tf.gfile.DeleteRecursively(save_files_to)
    tf.gfile.MakeDirs(save_files_to)

    X_train, X_test, y_train, y_test = load_exchange(FLAGS.data_dir + FLAGS.exchange, y_loc=True)

    weights = train(X_train, X_test, y_train, y_test, save_files_to)
    print (weights)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_steps', type=int, default=1000,
                      help='Number of steps to run trainer.')

    parser.add_argument('--n_step', type=int, default=100,
                      help='Number of steps to run trainer.')

    parser.add_argument('--learning_rate', type=float, default=0.01,
                      help='Initial learning rate')

    parser.add_argument('--data_dir', type=str, default='Exchanges/',
                      help='Summaries log directory')

    parser.add_argument('--exchange', type=str, default='RPOA')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)