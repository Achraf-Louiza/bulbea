from __future__ import absolute_import
from six import with_metaclass

from keras.models import Sequential
from keras.layers import LSTM, GRU, SimpleRNN, Dense, Activation, Dropout
from bulbea.learn.models import Supervised

class ANN(Supervised):
    pass

class RNNCell(object):
    RNN  = SimpleRNN
    GRU  = GRU
    LSTM = LSTM

class RNN(ANN):
    def __init__(self, sizes,
                 cell       = RNNCell.LSTM,
                 dropout    = 0.2,
                 activation = 'linear',
                 loss       = 'mse',
                 optimizer  = 'rmsprop'):
        self.model = Sequential()
        self.model.add(cell(
            units           = sizes[1],  # 'units' instead of 'output_dim'
            input_shape     = (None, sizes[0]),  # input_shape should be (timesteps, features)
            return_sequences = True
        ))

        for i in range(2, len(sizes) - 1):
            self.model.add(cell(units=sizes[i], return_sequences=True))
            self.model.add(Dropout(dropout))

        self.model.add(Dense(units=sizes[-1]))  # 'units' instead of 'output_dim'
        self.model.add(Activation(activation))

        self.model.compile(loss=loss, optimizer=optimizer)

    def fit(self, X, y, *args, **kwargs):
        return self.model.fit(X, y, *args, **kwargs)

    def predict(self, X):
        return self.model.predict(X)
