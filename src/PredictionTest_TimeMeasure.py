import tensorflow as tf
import numpy as np
import csv
import time
from time import strftime

start_time = time.time()

#weights = {}
#biases = {}

xx = np.empty([0,500,90],float)
f ='test9.csv'
data = [[ float(elm) for elm in v] for v in csv.reader(open(f, "r"))]
tmp1 = np.array(data)
xx = np.dstack(tmp1.T)
x2 = np.empty([0,500,90],float)
x2 = np.concatenate((x2, xx),axis=0)
saver = tf.train.import_meta_graph('model.ckpt.meta')
graph = tf.get_default_graph()
#weights['out'] = graph.get_tensor_by_name("Variable:0")
#biases['out'] = graph.get_tensor_by_name("Variable_1:0")
x = graph.get_tensor_by_name("Placeholder:0")
pred = graph.get_tensor_by_name("add:0")
#init = tf.global_variables_initializer()
with tf.Session() as sess:
    saver.restore(sess, 'model.ckpt')
    #sess.run(init)
    print("start_time", start_time)  # 출력해보면, 시간형식이 사람이 읽기 힘든 일련번호형식입니다.
    print("--- %s seconds ---" % (time.time() - start_time))
    now = strftime("%y%m%d-%H%M%S")
    print(now)

    start_time2 = time.time()

    n = pred.eval(feed_dict={x: x2})
    print(n)
    n2 = tf.argmax(n, 1)
    print(n2.eval())
    print("start_time2", start_time2)  # 출력해보면, 시간형식이 사람이 읽기 힘든 일련번호형식입니다.
    print("--- %s seconds ---" % (time.time() - start_time2))
    now2 = strftime("%y%m%d-%H%M%S")
    print(now2)

    start_time3 = time.time()

    n_1 = pred.eval(feed_dict={x: x2})
    print(n_1)
    n2_1 = tf.argmax(n_1, 1)
    print(n2_1.eval())
    print("start_time2", start_time3)  # 출력해보면, 시간형식이 사람이 읽기 힘든 일련번호형식입니다.
    print("--- %s seconds ---" % (time.time() - start_time3))
    now3 = strftime("%y%m%d-%H%M%S")
    print(now3)
    sess.close()
