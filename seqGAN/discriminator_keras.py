import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import core, Dense, Input, LSTM, Embedding, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, Multiply, Add, Lambda
from keras.backend import K
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers


# An alternative to tf.nn.rnn_cell._linear function, which has been removed in Tensorfow 1.0.1
# The highway layer is borrowed from https://github.com/mkroutikov/tf-lstm-char-cnn
def linear(input_, output_size, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * input_[i] ) + Bias[k]
    Args:
      input_: a tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(input_[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    #
    # Now the computation.
    # ----------------------------------------------------------------------------
    matrix = K.placeholder([output_size, input_size], dtype=input_.dtype)
    bias_term = K.placeholder([output_size], dtype=input_.dtype)
    return K.dot(input_, K.transpose(matrix)) + bias_term


def highway_layers(value, n_layers, activation="tanh", gate_bias=-3):
    dim = K.int_shape(value)[-1]
    gate_bias_initializer = keras.initializers.Constant(gate_bias)
    for i in range(n_layers):
        gate = Dense(units=dim, bias_initializer=gate_bias_initializer)(value)
        gate = Activation("sigmoid")(gate)
        negated_gate = Lambda(
            lambda x: 1.0 - x,
            output_shape=(dim,))(gate)
        transformed = Dense(units=dim, activation=activation)(value)
        transformed_gated = Multiply()([gate, transformed])
        identity_gated = Multiply()([negated_gate, value])
        value = Add()([transformed_gated, identity_gated])
    return value


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """
    for idx in range(num_layers):
        g = f(linear(input_, size, scope='highway_lin_%d' % idx))


    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output


class Discriminator(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    Args:
      sequence_length: the length of a sequence
      num_classes: the dimensionality of output vector, i.e., number of classes
      vocab_size: the number of vocabularies
      embedding_size: the dimensionality of an embedding vector
      filter_sizes: filter size
      num_filters: number of filter
      l2_reg_lambda: lambda, a parameter for L2 regularizer
    """

    def __init__(self, max_sequence_length, num_classes, vocab_size,
                 embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        #
        # Placeholders for input, output and dropout
        # ----------------------------------------------------------------------------
        self.input_x = Input(shape=(max_sequence_length,), dtype='int32')
        self.input_y = Input(shape=(num_classes,), dtype='float32')
        self.dropout_keep_prob = Input(shape=1, dtype='float32')


        # Keeping track of l2 regularization loss (optional)
        #l2_loss = tf.constant(0.0)

        #
        # Embedding layer
        # ----------------------------------------------------------------------------
        self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
        self.W = Embedding(vocab_size,
                           embedding_size,
                           weights=[tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0)],
                           input_length=max_sequence_length,
                           trainable=False)
        self.embedded_chars = self.W(self.input)
        self.embedded_chars_expanded = K.expand_dims(self.embedded_chars, -1)

        #
        # Create a convolution + maxpool layer for each filter size
        # ----------------------------------------------------------------------------
        pooled_outputs = []
        for filter_size, num_filter in zip(filter_sizes, num_filters):
            # Convolution Layer
            conv = Conv2D(filters=num_filter,
                          kernel_size=filter_size,
                          padding='valid',
                          activation="relu",
                          strides=1)(self.embedded_chars_expanded)
            # Max-pooling over the outputs
            conv = MaxPooling2D(pool_size=2)(conv)
            pooled_outputs.append(conv)

        #
        # Combine all the pooled features
        # ----------------------------------------------------------------------------
        num_filters_total = sum(num_filters)
        self.h_pool = concatenate(pooled_outputs, 3)
        self.h_pool_flat = K.reshape(self.h_pool, [-1, num_filters_total])

        #
        # Add highway
        # ----------------------------------------------------------------------------
        self.h_highway = keras.layers.core.Hi
        #highway(self.h_pool_flat, self.h_pool_flat.get_shape()[1], 1, 0)

        #
        # Add dropout
        # ----------------------------------------------------------------------------
        self.h_drop = Dropout(self.dropout_keep_prob)(self.h_highway)

        #
        # Final (unnormalized) scores and predictions
        # ----------------------------------------------------------------------------
        self.scores = BatchNormalization()(self.h_drop)
        self.preds = Dense(num_classes,
                           activation='softmax',
                           kernel_regularizer=regularizers.l2(0.01),
                           activity_regularizer=regularizers.l1(0.01))(self.h_drop)

        #
        # Train the model
        # ----------------------------------------------------------------------------
        model = Model(inputs=self.input_x,
                      outputs=self.preds)
        model.compile(loss='binary_crossentropy',
                      optimizer='nadam',
                      metrics=['acc'])

