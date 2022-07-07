import tensorflow.python as tf
import tensorflow
import numpy as np


class BagGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self, x_col, y_col=None, batch_size=32, num_classes=None, shuffle=True):
        self.batch_size = batch_size
        self.indices = np.arange(x_col.shape[0])
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.x_col = x_col
        self.y_col = y_col
        self.on_epoch_end()

    def __len__(self):
        return self.x_col.shape[0] // self.batch_size

    def __getitem__(self, index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]

        X, y = self.__get_data(batch)
        return X, y

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __get_data(self, batch):
        X = np.where(self.x_col[batch].toarray() > 0, 1, 0).astype('float32')
        y = None

        #for i, id in enumerate(batch):
        #    X[i,] =  # logic
        #    y[i] =  # labels

        return X, y