# encoding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# this is data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
lr = 1e-3
iterations = 100000
batch_size = 128

n_inputs = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # time steps
n_hidden_unis = 128 # neurons in hidden layer
n_classes = 10

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
weights = {
    #(28, 28)
    'in': tf.Variable(tf.random_uniform([n_inputs, n_hidden_unis]), dtype=tf.float32),

    #(128,10)
    'out': tf.Variable(tf.random_normal([n_hidden_unis, n_classes]), dtype=tf.float32)
}

biases = {
    #(128,)
    'in': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_hidden_unis])),

    #(10,)
    'out': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_classes]))
}

def RNN(X, weights, biases):
    """
    :param X: (128 batch, 28 steps, 28 inputs) (N,T,D)
    :param weights:
    :param biases:
    :return:
    """
    # hidden layer for input to cell
    X = tf.reshape(X, [-1, n_inputs])  # X(N*T, D)
    X_in = tf.matmul(X, weights['in']) + biases['in'] #X_in(N*T, H)
    X_in = tf.reshape(X_in, [batch_size, n_steps, n_hidden_unis]) #X_in(N,T,H)

    # cell
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_unis, forget_bias=1.0, state_is_tuple=True)
    # lstm cell is divided into two parts (c_state,m_state)
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    # states (c_state, m_state) for the last timestep
    # output is a list for all timesteps
    # time_major表示X_in中timestep的维度是不是第一个维度（0） 如果不是，那么就设置为False

    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)

    # hidden layer for output as the final results

    # Method 1
    results = tf.matmul(states[1], weights['out']) + biases['out']

    # Method 2
    # outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    # results = tf.matmul(outputs[-1], weights['out']) + biases['out']

    return results

pred = RNN(x,weights, biases)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
train_op = tf.train.AdamOptimizer(lr).minimize(loss)

correct_pred = tf.equal(tf.argmax(pred, axis=1), tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * batch_size < iterations:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])

        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
                x: batch_xs,
                y: batch_ys
            }))
        step += 1

