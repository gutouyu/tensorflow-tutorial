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
    X = tf.reshape(X, [-1, n_inputs])
    X_in = tf.matmul(X, weights['in']) + biases['in']
    X_in = tf.reshape(X_in, [batch_size, n_steps, n_inputs])

    # cell
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_unis, forget_bias=1.0, state_is_tuple=True)
    # TODO: fnish this

    # hidden layer for output as the final results

    results = None
    return results

pred = RNN(x,weights, biases)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logis=pred))
train_op = tf.train.AdamOptimizer(lr).minimize(loss)

correct_pred = tf.equal(tf.argmax(pred, axis=1), tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(correct_pred)

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

