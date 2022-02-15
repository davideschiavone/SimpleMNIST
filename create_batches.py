import numpy as np
from keras.datasets import mnist
from datasetsequence import DataSetSequence

(train_X, train_y), (test_X, test_y) = mnist.load_data()
dss_train = DataSetSequence(train_X, train_y, 128,'Batches/train')
dss_train.createBatches()

dss_test = DataSetSequence(train_X, train_y, 128,'Batches/test')
dss_test.createBatches()