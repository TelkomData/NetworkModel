import tensorflow as tf
from tensorflow.python.framework import ops


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

def nn_layer(input_tensor, num_features, num_classes, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):

        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.truncated_normal([num_features, num_classes], stddev=0.1, seed=42))
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.truncated_normal([num_classes], stddev=0.1, seed=42))
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(input_tensor, weights) + biases
            variable_summaries(Wx_plus_b)
        with tf.name_scope('activations'):
            activations = tf.maximum(tf.minimum(act(Wx_plus_b),0.99999),.00001)
            variable_summaries(activations)
        return activations

def multi_layer_nn(X, Y, model_name, levels = {}, learning_rate=0.1):

    num_features = X.shape[1]
    if len(Y.shape) == 1:
        num_classes = 1
    else:
        num_classes = Y.shape[1]
    previous_nodes = num_features

    with tf.name_scope(model_name):
        X = tf.placeholder(tf.float32, [None, num_features])
        Y = tf.placeholder(tf.float32, [None, num_classes])
        
        for k, v in levels.items():
            v = {'nodes': v}
            v[model] = nn_layer(X, previous_nodes, v['nodes'], k, tf.nn.relu)
            previous_nodes = v['nodes']

        if num_classes == 1:
            y_hat = nn_layer(level1, previous, num_classes, model_name, tf.sigmoid)
            cost = tf.reduce_mean(-Y * tf.log(y_hat) - (1-Y) * tf.log(1-y_hat))
            pred = tf.equal(Y, tf.cast(y_hat>0.5, tf.float32))        
        else:
            y_hat = nn_layer(X, num_features, num_classes, model_name, tf.identity)
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=Y))
            pred = tf.equal(tf.argmax(Y, 1), tf.argmax(y_hat, 1))

        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))
        return train_step, cost, accuracy

def train(X_train, X_test, y_train, y_test, model_name, levels = {}, learning_rate=0.1, n_step = 100, max_steps = 1000, save_file=None): 

    ops.reset_default_graph()
    
#    train_step, cost, accuracy = multi_layer_nn(X_train, y_train, model_name, levels, learning_rate)
    num_features = X_train.shape[1]
    if len(y_train.shape) == 1:
        num_classes = 1
    else:
        num_classes = y_train.shape[1]
    previous_nodes = num_features

#    with tf.name_scope(model_name):
    X = tf.placeholder(tf.float32, [None, num_features])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    
    for k, v in levels.items():
        v = {'nodes': v}
        v[model] = nn_layer(X, previous_nodes, v['nodes'], k, tf.nn.relu)
        previous_nodes = v['nodes']

    if num_classes == 1:
        y_hat = nn_layer(level1, previous, num_classes, 'final', tf.sigmoid)
        cost = tf.reduce_mean(-Y * tf.log(y_hat) - (1-Y) * tf.log(1-y_hat))
        pred = tf.equal(Y, tf.cast(y_hat>0.5, tf.float32))        
    else:
        y_hat = nn_layer(X, num_features, num_classes, 'final', tf.identity)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=Y))
        pred = tf.equal(tf.argmax(Y, 1), tf.argmax(y_hat, 1))

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))
#    return train_step, cost, accuracy

    tf.summary.scalar('cost', cost)
    tf.summary.scalar('accuracy', accuracy) 
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # create the summary logs
    if save_file != None:
        merged = tf.summary.merge_all()
        test_writer = tf.summary.FileWriter(save_file + 'model', sess.graph)
        train_writer = tf.summary.FileWriter(save_file + 'test')

    # save the model
    saver = tf.train.Saver()

    for i in range(max_steps):   
        
        train, summary = sess.run([train_step, merged], feed_dict={X:X_train[i*n_step:(i+1)*n_step], Y:y_train[i*n_step:(i+1)*n_step]})
        train_writer.add_summary(summary,i)

        if i % 100 == 0:
            acc, summary = sess.run([accuracy, merged], feed_dict={X:X_test, Y:y_test})
            test_writer.add_summary(summary, i)
            print('Accuracy at step {:d}: {:0.2f}%'.format(i, acc*100))

        if (i * n_step) > len(X_train):
            break

    if save_file:
        save_path = saver.save(sess, save_file+model_name+'.ckpt')

    acc, summary = sess.run([accuracy, merged], feed_dict={X:X_test, Y:y_test})
    test_writer.add_summary(summary, i)
    print('Accuracy at step {:d}: {:0.2f}%'.format(i, acc*100))

    sess.close()

    return acc
