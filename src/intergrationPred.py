import tensorflow as tf
import numpy as np

import matlab.engine
import numpy as np

eng = matlab.engine.start_matlab()
ARR_FINAL = np.empty([0,90], float)
k = 1; #%반복(iteration)을 위한 초기화
t = 0; #%특정 부분부터 잘라서 가져오고 싶을 때, 최초시작지점 선택
csi_trace = eng.read_bf_file('C:/Users/user/Documents/data/walk/walk_04/2018_05_09_walk10_04_delay1000.dat')
saver = tf.train.import_meta_graph('../learning2/model.ckpt.meta')
graph = tf.get_default_graph()
x = graph.get_tensor_by_name("Placeholder:0")
pred = graph.get_tensor_by_name("add:0")

#시간
while(k <= 500):
    csi_entry = csi_trace[t]
    csi = eng.get_scaled_csi(csi_entry)
    A = eng.abs(csi)
    ARR_OUT = np.empty([0], float)

    # ARR_1 = []
    # ARR_2 = []
    # ARR_3 = []
    ARR_OUT = np.concatenate((ARR_OUT, A[0][0]), axis=0)
    ARR_OUT = np.concatenate((ARR_OUT, A[0][1]), axis=0)
    ARR_OUT = np.concatenate((ARR_OUT, A[0][2]), axis=0)
    # ARR_3.append(A[0][2][:])
    # ARR_FINAL = [ARR_1 + ARR_2 + ARR_3] #% 합치기

    ARR_FINAL = np.vstack((ARR_FINAL, ARR_OUT))
    k = k + 1
    t = t + 1
#측정
xx = np.empty([1,500,90], float)
xx[0] = ARR_FINAL

with tf.Session() as sess:
    saver.restore(sess, '../learning2/model.ckpt')
    #sess.run(init)
    k=0
    count = 0
    #시간
    n = pred.eval(feed_dict={x: xx})
    n2 = tf.argmax(n, 1)
    result = n2.eval()
    #측정
    print("walk stand empty sit handup")
    print(result)

    sess.close()