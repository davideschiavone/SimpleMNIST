import numpy as np
import math

class DataSetSequence():

    def __init__(self, x_set, y_set, batch_size, file_name, create):
        if create:
            self.x, self.y = x_set, y_set
            self.batch_size = batch_size
            self.file_name  = file_name
        else:
            self.batch_size = batch_size
            self.file_name  = file_name

    def len(self):
        return math.ceil(len(self.x) / self.batch_size)

    def createBatches(self):
        for i in range(self.len()):
            batch_name = self.file_name+str(i)+'.npy'
            with open(batch_name, 'wb') as f:
                np.save(f, self.x[i:i+self.batch_size])
                np.save(f, self.y[i:i+self.batch_size])

    def getItem(self, idx):
        batch_name = self.file_name+str(idx)+'.npy'
        with open(batch_name, 'rb') as f:
            batch_x = np.load(f)
            batch_y = np.load(f)

        return batch_x, batch_y