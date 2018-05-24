import tensorflow as tf
import numpy as np
import csv
import time
# input으로 이름을 받아서 그 이름에 해당하는 파일을 읽도록 하는 함수 작성
# for문을 돌면서 0을 입력받으면 프로그램이 종료되게 작성
START = time.time()
xx = np.empty([0,500,90],float)
f ='2018_05_14_empty_13.csv'
data = [[ float(elm) for elm in v] for v in csv.reader(open(f, "r"))]
tmp1 = np.array(data)
saver = tf.train.import_meta_graph('model.ckpt.meta')
graph = tf.get_default_graph()
x = graph.get_tensor_by_name("Placeholder:0")
pred = graph.get_tensor_by_name("add:0")

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess: # tensorflow-gpu 사용.
    saver.restore(sess, 'model.ckpt')
    #sess.run(init)
    for i in range(0, len(tmp1), 200):
        if i + 500 >= len(tmp1):
            break
        xx = np.dstack(tmp1[i:500+i].T)
        n = pred.eval(feed_dict={x: xx})
        print(n)
        n2 = tf.argmax(n, 1)
        print(n2.eval())
        END = time.time() - START
        print("prediction time : ", END)
    sess.close()
