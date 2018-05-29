"""Functions for downloading and reading MNIST data."""
from __future__ import print_function
import gzip
import os
import numpy as np,numpy
import csv
import glob
import pandas as pd

class DataSet(object):
    def __init__(self, images, labels, fake_data=False):
        assert images.shape[0] == labels.shape[0], (
                "images.shape: %s labels.shape: %s" % (images.shape,
                                                        labels.shape))
        self._num_examples = images.shape[0]
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
    @property
    def images(self):
        return self._images
    @property
    def labels(self):
        return self._labels
    @property
    def num_examples(self):
        return self._num_examples
    @property
    def epochs_completed(self):
        return self._epochs_completed
    def next_batch(self, batch_size, fake_data=False):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

def csv_import():
    x_dic = {}
    y_dic = {}
    print("csv file importing...")

    for i in ["walk", "stand", "empty", "sitdown", "up"]: #modified


        SKIPROW = 2 #Skip every 2 rows -> overlap 800ms to 600ms  (To avoid memory error)
        num_lines = sum(1 for l in open("./input_files/xx_1000_60_" + str(i) + ".csv"))
        skip_idx = [x for x in range(1, num_lines) if x % SKIPROW !=0]

        xx = np.array(pd.read_csv("./input_files/xx_1000_60_" + str(i) + ".csv", header=None, skiprows = skip_idx))
        yy = np.array(pd.read_csv("./input_files/yy_1000_60_" + str(i) + ".csv", header=None, skiprows = skip_idx))


        xx = xx.reshape(len(xx),1000,90)

        # 1000 Hz to 500 Hz (To avoid memory error)
        xx = xx[:,::2,:90]

        x_dic[str(i)] = xx
        y_dic[str(i)] = yy

        print(str(i), "finished...", "xx=", xx.shape, "yy=",  yy.shape)

    return x_dic["walk"], x_dic["stand"], x_dic["empty"], x_dic["sitdown"], x_dic["up"],\
        y_dic["walk"], y_dic["stand"], y_dic["empty"], y_dic["sitdown"], y_dic["up"]
        # modified - might need to obmit up in the future