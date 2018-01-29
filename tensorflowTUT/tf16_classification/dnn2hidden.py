from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(r"C:\Users\dcw\Documents\Machine Learning\MNIST", one_hot=True)
# allows you further divide data
train_X = mnist.train.images
test_X = mnist.test.images

n_samples = train_X.shape[0]

import tensorflow as tf
# Parameters
learning_rate = .5
num_steps = 500
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 28 # 1st layer number of neurons
n_hidden_2 = 28 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)


def add_layer(inputs, in_size, out_size, activation_function=None,):
# Store layers weight & bias
    weights = {
    'h1': tf.Variable(tf.random_normal([in_size, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, out_size]))
    }
    biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([out_size]))
    }    
    # add one more layer and return the output of this layer
    
    if activation_function is None:
        layer_1 = tf.add(tf.matmul(inputs, weights['h1']), biases['b1'])
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        
    else:
        layer_1 = activation_function(tf.add(tf.matmul(inputs, weights['h1']), biases['b1']))
        layer_2 = activation_function(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
        out_layer = activation_function(tf.matmul(layer_2, weights['out']) + biases['out'])
        
    return out_layer

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, num_input]) # 28x28
ys = tf.placeholder(tf.float32, [None, num_classes])

# add output layer
prediction = add_layer(xs, num_input, num_classes,  activation_function=tf.nn.sigmoid) # dw: change from softmax 

# the error between prediction and real data

# cross entropy with softmax 88% (no hidden layers)
# cross entropy with softmax 43% (2 hidden layers)
# cross entropy with sigmoid 17% (no hidden layer)
# cross entropy with sigmoid 11% (2 hidden layers)
cost = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))       # loss
# Mean squared error
# mse with softmax 60% (no hidden layers)
# mse with softmax 24% (2 hidden layers)
# mse with sigmoid 29% (no hidden layers)
# mse with sigmoid 57% (2 hidden layers)
#cost = tf.reduce_sum(tf.pow(prediction-ys, 2))/(2*batch_size) # dw: should be batch size instead of n_samples, right?
# train operation
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

# train 1000 times
for i in range(1001):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(i, " iteration: ", compute_accuracy(
            mnist.test.images, mnist.test.labels))
        
