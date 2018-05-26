import tensorflow as tf
import numpy as np
import matlab.engine
import matplotlib.pyplot as plt
from drawnow import drawnow
def make_fig():
    plt.xlim(0, 15000)
    plt.ylim(0, 40)
    plt.plot(b, xx1, color='red')
    plt.plot(b, yy1, color='blue')
    plt.plot(b, zz1, color='yellow')

eng = matlab.engine.start_matlab()
saver = tf.train.import_meta_graph('../learning2/model.ckpt.meta')
graph = tf.get_default_graph()
x = graph.get_tensor_by_name("Placeholder:0")
pred = graph.get_tensor_by_name("add:0")
plt.ion()
plt.figure(0)
b = np.arange(0, 15000)
while 1:
    k = 1
    t = 0
    h = 0
    csi_trace = eng.read_bf_file('C:/Users/user/Documents/data/walk/walk_04/2018_05_09_walk10_04_delay1000.dat')
    ARR_FINAL = np.empty([0, 90], float)
    xx = np.empty([1, 500, 90], float)
    xx1 = np.empty([0], float)
    yy1 = np.empty([0], float)
    zz1 = np.empty([0], float)
    while (k <= 500):
        csi_entry = csi_trace[t]
        csi = eng.get_scaled_csi(csi_entry)
        A = eng.abs(csi)
        ARR_OUT = np.empty([0], float)

        ARR_OUT = np.concatenate((ARR_OUT, A[0][0]), axis=0)
        ARR_OUT = np.concatenate((ARR_OUT, A[0][1]), axis=0)
        ARR_OUT = np.concatenate((ARR_OUT, A[0][2]), axis=0)

        xx1 = np.concatenate((xx1, A[0][0]), axis = 0)
        yy1 = np.concatenate((yy1, A[0][1]), axis = 0)
        zz1 = np.concatenate((zz1, A[0][2]), axis = 0)
        ARR_FINAL = np.vstack((ARR_FINAL, ARR_OUT))
        k = k + 1
        t = t + 1
        h = h + 30
    xx[0] = ARR_FINAL
    drawnow(make_fig)

    with tf.Session() as sess:
        saver.restore(sess, '../learning2/model.ckpt')
        n = pred.eval(feed_dict={x: xx})
        n2 = tf.argmax(n, 1)
        result = n2.eval()
        print("walk stand empty sit handup")
        print(result)

        sess.close()