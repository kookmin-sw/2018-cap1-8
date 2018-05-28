import tensorflow as tf
import numpy as np
import matlab.engine
import matplotlib.pyplot as plt
from drawnow import drawnow
def make_fig():
    plt.xlim(0, 1000)
    plt.ylim(0, 40)
    plt.plot(b, xx1, color='red', alpha = 0.5)
    plt.plot(b, yy1, color='blue', alpha = 0.5)
    plt.plot(b, zz1, color='green', alpha = 0.5)
    plt.ylabel("Amplitude")
    if (result == 0):
        plt.title("Walk")
    elif (result == 1):
        plt.title("Stand")
    elif (result == 2):
        plt.title("Empty")
    elif (result == 3):
        plt.title("Sit down")
    elif (result == 4):
        plt.title("Stand up")
    else:
        plt.title("?")

eng = matlab.engine.start_matlab()
saver = tf.train.import_meta_graph('../model/model.ckpt.meta')
graph = tf.get_default_graph()
x = graph.get_tensor_by_name("Placeholder:0")
pred = graph.get_tensor_by_name("add:0")
plt.ion()
plt.figure(0)
b = np.arange(0, 15000)
act = ["Walk", "Stand", "Empty", "Sit down", "Stand up"]
while 1:
    k = 1
    t = 0
    csi_trace = eng.read_bf_file('/home/kjlee/linux-80211n-csitool-supplementary/netlink/test.dat')
    if len(csi_trace) < 500:
        continue
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
    xx[0] = ARR_FINAL

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        saver.restore(sess, '../model/model.ckpt')
        n = pred.eval(feed_dict={x: xx})
        n2 = tf.argmax(n, 1)
        result = n2.eval()
        drawnow(make_fig)
        print(act[int(result)])

        sess.close()