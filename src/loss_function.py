import tensorflow as tf
from tensorflow.keras.losses import Loss

class CustomLoss(Loss):
    def __init__(self, name="custom_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1, 1]), dtype=tf.float32)  # Ensure y_true is the same shape and type as y_pred
        y_pred = tf.cast(y_pred, dtype=tf.float32)  # Ensure y_pred is float32
        main_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        auxiliary_loss = self.calculate_auxiliary_loss(y_true, y_pred)
        total_loss = main_loss + auxiliary_loss
        return total_loss

    @tf.function
    def calculate_auxiliary_loss(self, y_true, y_pred):
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        return mse_loss

    def get_config(self):
        base_config = super(CustomLoss, self).get_config()
        return base_config
