import tensorflow as tf
import numpy as np
import csv


f ='Dataset2/test용/2018_05_14_handup03_01.csv'
data = [[ float(elm) for elm in v] for v in csv.reader(open(f, "r"))]
tmp1 = np.array(data)
slide_size = 200

saver = tf.train.import_meta_graph('learning2/model.ckpt.meta')
graph = tf.get_default_graph()
x = graph.get_tensor_by_name("Placeholder:0")
pred = graph.get_tensor_by_name("add:0")
arrMatrix = np.zeros((1,5))
with tf.Session() as sess:
    saver.restore(sess, 'learning2/model.ckpt')
    #sess.run(init)
    k=0
    count = 0
    while k <= (len(tmp1) + 1 - 2 * 500):
        x2 = np.empty([0, 500, 90], float)
        xx = np.dstack(np.array(tmp1[k:k + 500, 0:90]).T)
        x2 = np.concatenate((x2, xx), axis=0)
        k += slide_size

        n = pred.eval(feed_dict={x: x2})
        print(n)
        n2 = tf.argmax(n, 1)
        result = n2.eval()
        arrMatrix[0][result] += 1
        print(result)
        count += 1
    print()
    print("walk stand empty sit handup (%)")
    print((arrMatrix / count) * 100)

    sess.close()