from base.base_model import BaseModel
import tensorflow as tf


class NTNModel(BaseModel):
    def __init__(self, config):
        super(NTNModel, self).__init__(config)
        self.build_model()
        self.init_saver()

        self.loss = 0
        self.train_step = 0
        self.accuracy = 0

    def build_model(self):
        # here you build the tensorflow graph of any model you want and also define the loss.
        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=d2))
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cross_entropy,
                                                                                             global_step=self.global_step_tensor)
            correct_prediction = tf.equal(tf.argmax(d2, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        pass

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

