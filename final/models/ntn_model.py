from base.base_model import BaseModel

import tensorflow as tf

from tensorflow.keras.initializers import he_normal
from tensorflow.nn import tanh
from tensorflow.nn import relu

from utils.utils import assert_shape


class NTNModel(BaseModel):
    def __init__(self, config):
        super(NTNModel, self).__init__(config)
        self.build_model()
        self.init_saver()

        self.loss = 0
        self.accuracy = 0

    def build_model(self):
        # Dimensions
        self.is_training = tf.placeholder(tf.bool)
        # self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        # self.y = tf.placeholder(tf.float32, shape=[None, 10])

        W = tf.get_variable('W', shape=[k, 2*d], initializer=he_normal())
        b = tf.get_variable('b', shape=[k], initializer=tf.zeros_initializer())
        T_1 = tf.get_variable('T_1', shape=[d, d, k], initializer=he_normal())
        T_2 = tf.get_variable('T_2', shape=[d, d, k], initializer=he_normal())
        T_3 = tf.get_variable('T_3', shape=[d, d, k], initializer=he_normal())

        # Architecture
        R_1 = self.bilinear_product(O_1, T_1, P)
        assert_shape(R_1, k)

        R_2 = self.bilinear_product(P, T_2, O_2)
        assert_shape(R_2, k)

        U = self.bilinear_product(R_1, T_3, R_2)
        assert_shape(U, k)

        # Loss and accuracy
        with tf.name_scope("loss"):

            self.loss = relu(1 - tanh(e) + tanh(e_r))  # FIXME: + tf.nn.l2_normalize(theta)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss,
                                                                                             global_step=self.global_step_tensor)
            correct_prediction = tf.equal(tf.argmax(d2, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        pass

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def bilinear_product(self, E1, t2, E3):
        # Reference https://stackoverflow.com/a/34113467
        temp = tf.matmul(E1, tf.reshape(Wddk, [d, d * k]))
        return tf.matmul(E2, tf.reshape(temp, [d, k]))

