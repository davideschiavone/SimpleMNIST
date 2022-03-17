import numpy as np
from keras.datasets import mnist
from datasetsequence import DataSetSequence

(train_X, train_y), (test_X, test_y) = mnist.load_data()
dss_train = DataSetSequence(train_X, train_y, 256,'Batches/train', True)
dss_train.createBatches()

dss_test = DataSetSequence(test_X, test_y, 256,'Batches/test', True)
dss_test.createBatches()