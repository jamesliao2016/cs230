from base.base_model import BaseModel

import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.nn import tanh
from tensorflow.nn import relu

from utils.utils import assert_shape


class DenseModel(BaseModel):
    def __init__(self, config):
        super(NTNModel, self).__init__(config)
        self.build_model()
        self.init_saver()

        self.loss = 0
        self.accuracy = 0

    def build_model(self):
        '''

        INPUT Emb(S) ->
        NN knowledge of learned features (FNN/CNN/RNN?) ->
        tanh (impact) ->
        reduce sum ->
        OUTPUT sigmoid (market increase yes or no)

        :return:
        '''

        # as first layer in a sequential model:
        model = Sequential()
        model.add(Dense(100, input_shape=(512,), activation='relu'))
        model.add(Dense(1, activation='sigmoid'))


    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
