import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def add_layer(n_layer, inputs, in_size, out_size, activation_function=None):

    with tf.name_scope('layer%d' % n_layer):

        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram('/weights', Weights)

        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1, name='b')
            tf.summary.histogram('/biases', biases)

        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights) , biases)
            tf.summary.histogram('Wx_plus_b', Wx_plus_b)

        if activation_function is None:
            output = Wx_plus_b
        else:
            output = activation_function(Wx_plus_b)

        tf.summary.histogram('/output', output)
        return output

def compute_accuracy(v_xs, v_ys):
	global prediction
	y_pred = sess.run(prediction, feed_dict={xs:v_xs})
	correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(v_ys,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys})
	return result



# Define palceholder
xs = tf.placeholder(tf.float32, [None,28*28]) #784
ys = tf.placeholder(tf.float32, [None,10])

# Add hidden layer
hidden_size = 100
layer1 = add_layer(1, xs, 784, hidden_size, activation_function=tf.nn.sigmoid)

# Add output layer
prediction = add_layer(2, layer1, hidden_size,10, activation_function=tf.nn.softmax)

# The error between prediction and real data
cross_entropy = tf.reduce_mean(
	-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(1).minimize(cross_entropy)

sess = tf.Session()
#important step
sess.run(tf.global_variables_initializer())

for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step,feed_dict={xs:batch_xs, ys:batch_ys})

	if i % 50 == 0:
		print(compute_accuracy(mnist.test.images, mnist.test.labels), compute_accuracy(batch_xs, batch_ys))











































