import tensorflow as tf
import numpy as np
import matlab.engine
import numpy as np

eng = matlab.engine.start_matlab()
saver = tf.train.import_meta_graph('../learning2/model.ckpt.meta')
graph = tf.get_default_graph()
x = graph.get_tensor_by_name("Placeholder:0")
pred = graph.get_tensor_by_name("add:0")

while 1:
    k = 1
    t = 0
    csi_trace = eng.read_bf_file('C:/Users/user/Documents/data/walk/walk_04/2018_05_09_walk10_04_delay1000.dat')
    ARR_FINAL = np.empty([0, 90], float)
    xx = np.empty([1, 500, 90], float)
    while (k <= 500):
        csi_entry = csi_trace[t]
        csi = eng.get_scaled_csi(csi_entry)
        A = eng.abs(csi)
        ARR_OUT = np.empty([0], float)

        ARR_OUT = np.concatenate((ARR_OUT, A[0][0]), axis=0)
        ARR_OUT = np.concatenate((ARR_OUT, A[0][1]), axis=0)
        ARR_OUT = np.concatenate((ARR_OUT, A[0][2]), axis=0)

        ARR_FINAL = np.vstack((ARR_FINAL, ARR_OUT))
        k = k + 1
        t = t + 1
    xx[0] = ARR_FINAL

    with tf.Session() as sess:
        saver.restore(sess, '../learning2/model.ckpt')
        n = pred.eval(feed_dict={x: xx})
        n2 = tf.argmax(n, 1)
        result = n2.eval()
        print("walk stand empty sit handup")
        print(result)

        sess.close()