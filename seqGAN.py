#
# The code is modified from https://github.com/berleon/seqgan/blob/master/seqgan/seqgan.py
#

from keras.engine.training import Model
#from keras.engine.topology import Merge
import numpy as np
import keras.backend as K
from contextlib import contextmanager
import keras.callbacks as cbks

from keras.layers.merge import concatenate

TODO = "todo"


@contextmanager
def trainable(model, trainable):
    trainables = []
    for layer in model.layers:
        trainables.append(layer.trainable)
        layer.trainable = trainable
    yield
    for t, layer in zip(trainables, model.layers):
        layer.trainable = t


def prob_to_sentence(prob):
    fake_idx = np.argmax(prob, axis=-1)
    fake = np.zeros_like(prob)
    fake[:, :, fake_idx] = 1
    return fake


class SeqGAN:

    #
    # Initialization
    # ----------------------------------------------------------------------------
    def __init__(self, g, d, m, g_optimizer, d_optimizer):

        # Model of generator
        self.g = g

        # Model of discriminator
        self.d = d

        # Model of ???
        self.m = m

        self.z, self.seq_input = self.g.inputs
        self.fake_prob, = self.g.outputs

        with trainable(m, False):
            #m_input = merge([self.seq_input, self.fake_prob], mode='concat', concat_axis=1)
            m_input = concatenate([self.seq_input, self.fake_prob], axis=1)
            self.m_realness = self.m(m_input)
            self.model_fit_g = Model(inputs=[self.z, self.seq_input],
                                     outputs=[self.m_realness])
            self.model_fit_g.compile(optimizer=g_optimizer,
                                     loss=K.binary_crossentropy)

        self.d.compile(optimizer=d_optimizer,
                       loss=K.binary_crossentropy)

    #
    # return the shape of input noise variables z
    # ----------------------------------------------------------------------------
    def z_shape(self, batch_size=64):
        layer, _, _ = self.z._keras_history
        return (batch_size,) + layer.output_shape[1:]

    #
    # Sample input noise variables z
    # ----------------------------------------------------------------------------
    def sample_z(self, batch_size=64):
        shape = self.z_shape(batch_size)
        return np.random.uniform(-1, 1, shape)

    #
    # Generate the fake samples with: the input noise variables z & input sequence seq_input
    # ----------------------------------------------------------------------------
    def generate(self, z, seq_input, batch_size=32):
        return self.g.predict([z, seq_input], batch_size=batch_size)

    #
    # Training
    # ----------------------------------------------------------------------------
    def train_on_batch(self, seq_input, real, d_target=None):
        nb_real = len(real)
        nb_fake = len(seq_input)
        if d_target is None:
            d_target = np.concatenate([
                np.zeros((nb_fake, 1)),
                np.ones((nb_real, 1))
            ])
        fake_prob = self.generate(self.sample_z(nb_fake), seq_input)
        fake = np.concatenate([seq_input, prob_to_sentence(fake_prob)], axis=1)
        fake_and_real = np.concatenate([fake, real], axis=0)
        d_loss = self.d.train_on_batch(fake_and_real, d_target)
        d_realness = self.d.predict(fake)
        m_loss = self.m.train_on_batch(np.concatenate([seq_input, fake_prob], axis=1), d_realness)
        g_loss = self.model_fit_g.train_on_batch([self.sample_z(nb_fake), seq_input],
                                                 np.ones((nb_fake, 1)))
        return g_loss, d_loss, m_loss

    #
    # Training
    # ----------------------------------------------------------------------------
    def fit_generator(self, generator, nb_epoch, nb_batches_per_epoch, callbacks=[],
                      batch_size=None,
                      verbose=False):
        if batch_size is None:
            batch_size = 2*len(next(generator)[0])

        out_labels = ['g', 'd', 'm']

        self.history = cbks.History()
        callbacks = [cbks.BaseLogger()] + callbacks + [self.history]
        if verbose:
            callbacks += [cbks.ProgbarLogger()]
        callbacks = cbks.CallbackList(callbacks)
        callbacks._set_model(self)
        callbacks._set_params({
            'nb_epoch': nb_epoch,
            'nb_sample': nb_batches_per_epoch*batch_size,
            'verbose': verbose,
            'metrics': out_labels,
        })
        callbacks.on_train_begin()

        for e in range(nb_epoch):
            callbacks.on_epoch_begin(e)
            for batch_index, (seq_input, real) in enumerate(generator):
                callbacks.on_batch_begin(batch_index)
                batch_logs = {}
                batch_logs['batch'] = batch_index
                batch_logs['size'] = len(real) + len(seq_input
                                                     )
                outs = self.train_on_batch(seq_input, real)

                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o

                callbacks.on_batch_end(batch_index, batch_logs)
                if batch_index + 1 == nb_batches_per_epoch:
                    break

            callbacks.on_epoch_end(e)
        callbacks.on_train_end()
