import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

# Load data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.3)

def add_layer(n_layer, inputs, in_size, out_size, activation_function=None):

    with tf.name_scope('layer%d' % n_layer):

        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            # tf.summary.histogram('/weights', Weights)

        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1, name='b')
            # tf.summary.histogram('/biases', biases)

        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights) , biases)

            # Dropout
            Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)

            # tf.summary.histogram('Wx_plus_b', Wx_plus_b)

        if activation_function is None:
            output = Wx_plus_b
        else:
            output = activation_function(Wx_plus_b)

        # tf.summary.histogram('/output', output)
        return output

def compute_accuracy(v_xs, v_ys, dropout_rate):
	global prediction
	y_pred = sess.run(prediction, feed_dict={xs:v_xs, keep_prob: dropout_rate})
	correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(v_ys,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys, keep_prob: dropout_rate})
	return result



# Define palceholder
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None,64])
ys = tf.placeholder(tf.float32, [None,10])

# Add hidden layer
hidden_size = 100
layer1 = add_layer(1, xs, 64, hidden_size, activation_function=tf.nn.tanh)

# Add output layer
prediction = add_layer(2, layer1, hidden_size,10, activation_function=tf.nn.softmax)

# The error between prediction and real data
cross_entropy = tf.reduce_mean(
	-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))

tf.summary.scalar('loss', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

sess = tf.Session()

# Merge summary
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./logs/train', sess.graph)
test_writer = tf.summary.FileWriter('./logs/test', sess.graph)


#important step
sess.run(tf.global_variables_initializer())

for i in range(500):
	sess.run(train_step,feed_dict={xs:X_train, ys:y_train, keep_prob:0.5})
	if i % 50 == 0:
            train_result=sess.run(merged, feed_dict={xs:X_train, ys:y_train, keep_prob: 1.0})
            test_result=sess.run(merged, feed_dict={xs:X_test, ys:y_test, keep_prob: 1.0})
            train_writer.add_summary(train_result,i)
            test_writer.add_summary(test_result, i)
            print('Train_accu, Test_accu, %.2f%%, %.2f%%') % (compute_accuracy(X_train, y_train, 1.0) * 100, compute_accuracy(X_test, y_test, 1.0) * 100)

