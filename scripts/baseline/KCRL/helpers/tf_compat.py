import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import distributions as distr
from tensorflow.compat.v1.nn.rnn_cell import (
    DropoutWrapper,
    GRUCell,
    LSTMCell,
    MultiRNNCell,
)


tf.disable_v2_behavior()


class _LayersCompat:
    @staticmethod
    def dense(inputs, units, activation=None, kernel_initializer=None, use_bias=True, name=None):
        layer = tf.keras.layers.Dense(
            units=units,
            activation=activation,
            kernel_initializer=kernel_initializer,
            use_bias=use_bias,
            name=name,
        )
        return layer(inputs)

    @staticmethod
    def conv1d(inputs, filters, kernel_size, activation=None, use_bias=True, name=None, **kwargs):
        layer = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            use_bias=use_bias,
            name=name,
            **kwargs,
        )
        return layer(inputs)

    @staticmethod
    def dropout(inputs, rate, training=False, name=None, **kwargs):
        if rate == 0 or rate == 0.0:
            return inputs
        if isinstance(training, bool):
            return tf.nn.dropout(inputs, rate=rate, name=name) if training else inputs
        training = tf.cast(training, tf.bool)
        return tf.cond(
            training,
            lambda: tf.nn.dropout(inputs, rate=rate, name=name),
            lambda: tf.identity(inputs),
        )

    @staticmethod
    def batch_normalization(inputs, axis=-1, training=False, name=None, reuse=None, **kwargs):
        layer = tf.keras.layers.BatchNormalization(axis=axis, name=name, **kwargs)
        return layer(inputs, training=training)


tf.layers = _LayersCompat()


def xavier_initializer():
    return tf.keras.initializers.glorot_uniform()


def bias_add(value, scope="bias_add"):
    channels = value.get_shape().as_list()[-1]
    with tf.variable_scope(None, default_name=scope):
        bias = tf.get_variable(
            "bias",
            shape=[channels],
            initializer=tf.zeros_initializer(),
        )
    return tf.nn.bias_add(value, bias)
