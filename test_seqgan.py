from seqgan.seqgan import SeqGAN
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.engine.training import Model
from keras.optimizers import Adam

from keras.engine.topology import Input, merge
from beras.util import sequential

import pytest
import numpy as np

output_size = 64
input_size = 64
z_size = 48
nb_chars = 32


@pytest.fixture
def g():
    seq = Input(shape=(input_size, nb_chars))
    z = Input(shape=(z_size,))
    z_rep = RepeatVector(input_size)(z)
    seq_and_z = merge([seq, z_rep], mode='concat', concat_axis=-1)
    fake_prob = sequential([
        LSTM(8),
        RepeatVector(output_size),
        LSTM(8, return_sequences=True),
        TimeDistributed(Dense(nb_chars, activation='softmax')),
    ])(seq_and_z)

    g = Model([z, seq], [fake_prob])
    return g


@pytest.fixture
def d():
    x = Input(shape=(input_size + output_size, nb_chars))
    d_realness = sequential([
        LSTM(100),
        Dense(1, activation='sigmoid'),
    ])(x)
    d = Model([x], [d_realness])
    return d


@pytest.fixture
def m():
    x = Input(shape=(input_size + output_size, nb_chars))
    m_realness = sequential([
        LSTM(14),
        Dense(1, activation='sigmoid'),
    ])(x)
    m = Model([x], [m_realness])
    m.compile(Adam(), 'mse')
    return m

batch_size = 10


@pytest.fixture
def seq_one_hot():
    seq_one_hot = np.zeros((batch_size, input_size, nb_chars))
    for j in range(len(seq_one_hot)):
        seq_chars = np.random.randint(0, nb_chars, (input_size,))
        for i, char in enumerate(seq_chars):
            seq_one_hot[j, i, char] = 1
    return seq_one_hot


def test_seqgan_z_shape(g, d, m):
    gan = SeqGAN(g, d, m, Adam(), Adam())
    assert gan.z_shape(batch_size=128) == (128, z_size)


def test_seqgan_sample_z(g, d, m):
    gan = SeqGAN(g, d, m, Adam(), Adam())
    assert gan.sample_z(batch_size=128).shape == (128, z_size)


def test_model_d(d, seq_one_hot):
    d.compile(Adam(), 'binary_crossentropy')
    input = np.concatenate([seq_one_hot, seq_one_hot], axis=1)
    d.predict(input)
    d.train_on_batch(input, np.ones((batch_size, 1)))


def test_seqgan_train_on_batch(g, d, m, seq_one_hot):
    gan = SeqGAN(g, d, m, Adam(), Adam())
    real = np.concatenate([seq_one_hot, seq_one_hot], axis=1)
    losses = gan.train_on_batch(seq_one_hot, real)
    print(losses)


def test_seqgan_fit_generator(g, d, m, seq_one_hot):
    def generator():
        while True:
            fake_seed = seq_one_hot
            real = np.concatenate([seq_one_hot, seq_one_hot], axis=1)
            yield fake_seed, real

    gan = SeqGAN(g, d, m, Adam(), Adam())
    gan.fit_generator(generator(), 3, 20, verbose=True)
