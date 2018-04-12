import tensorflow as tf
import numpy as np
import csv
from tensorflow.contrib import rnn
window_size = 500
threshold = 60

# Network Parameters
n_input = 90 # WiFi activity data input (img shape: 90*window_size)
n_steps = window_size # timesteps
n_hidden = 200 # hidden layer num of features original 200
n_classes = 7 # WiFi activity total classes
x = tf.placeholder("float", [None, n_steps, n_input]) #500행(윈도우사이즈) 90열 정해지지 않은 층의 3차원 입력값
y = tf.placeholder("float", [None, n_classes]) #7열 정해지지 않은 행의 2차원 입력값

weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(x, n_steps, 0)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

xx = np.empty([0,500,90],float)
f ='test7.csv'
data = [[ float(elm) for elm in v] for v in csv.reader(open(f, "r"))]
tmp1 = np.array(data)
xx = np.dstack(tmp1.T)
#index = 0
x2 =np.empty([0,window_size,90],float)
x2 = np.concatenate((x2, xx),axis=0)
pred = RNN(x, weights, biases)
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('model.ckpt.meta')
    saver.restore(sess, 'model.ckpt')
    init = tf.global_variables_initializer()
    sess.run(init)
    n = pred.eval(feed_dict={x: x2})
    print(n)
    n2 = tf.argmax(n, 1)
    n3 = n2.eval()
    print(n3)
    sess.close()