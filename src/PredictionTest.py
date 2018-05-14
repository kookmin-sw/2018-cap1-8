import tensorflow as tf
import numpy as np
import csv

xx = np.empty([0,500,90],float)
f ='test/testempty.csv'
data = [[ float(elm) for elm in v] for v in csv.reader(open(f, "r"))]
tmp1 = np.array(data)
xx = np.dstack(tmp1.T)
x2 = np.empty([0,500,90],float)
x2 = np.concatenate((x2, xx),axis=0)
saver = tf.train.import_meta_graph('LR0.0001_BATCHSIZE200_NHIDDEN200/model.ckpt.meta')
graph = tf.get_default_graph()
x = graph.get_tensor_by_name("Placeholder:0")
pred = graph.get_tensor_by_name("add:0")

with tf.Session() as sess:
    saver.restore(sess, 'LR0.0001_BATCHSIZE200_NHIDDEN200/model.ckpt')
    #sess.run(init)
    n = pred.eval(feed_dict={x: x2})
    print(n)
    n2 = tf.argmax(n, 1)
    print(n2.eval())
    sess.close()