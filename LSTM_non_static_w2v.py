'''
Single model may achieve LB scores at around 0.29+ ~ 0.30+
Average ensembles can easily get 0.28+ or less
Don't need to be an expert of feature engineering
All you need is a GPU!!!!!!!
'''


#
# Import packages
# ----------------------------------------------------------------------------
import os
import re
import csv
import codecs
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

import sys
from importlib import reload

import preprocessing

reload(sys)
# Since the default on Python 3 is UTF-8 already, there is no point in leaving those statements in.
# sys.setdefaultencoding('utf-8')


#
# Set directories and parameters
# ----------------------------------------------------------------------------
test_CVID = 0

BASE_DIR = '../input/'
EMBEDDING_FILE = '/home/csist/workspace/resources/GoogleNews-vectors-negative300.bin'
TRAIN_DATA_FILE = '/home/csist/Dataset/QuoraQP/CV-' + str(test_CVID) + '_clean.csv'
VALID_DATA_FILE = '/home/csist/Dataset/QuoraQP/CV' + str(test_CVID) + '_clean.csv'
TEST_DATA_FILE = '/home/csist/Dataset/QuoraQP/test_clean.csv'

#
# Set directories to small test data
# ----------------------------------------------------------------------------
# EMBEDDING_FILE = '/data1/resources/GoogleNews-vectors-negative300.bin'
# TRAIN_DATA_FILE = '/data1/quora_pair/50q_pair.csv'
# VALID_DATA_FILE = '/data1/quora_pair/50q_pair.csv'
# TEST_DATA_FILE = '/data1/quora_pair/50q_pair_test.csv'

MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

num_lstm = np.random.randint(175, 275)
num_dense = np.random.randint(100, 150)
rate_drop_lstm = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25

act = 'relu'
re_weight = True  # whether to re-weight classes to fit the 17.5% share in test set

STAMP = 'nslstm_%d_%d_%.2f_%.2f' % (num_lstm, num_dense, rate_drop_lstm, rate_drop_dense)


#
# Index word vectors
# ----------------------------------------------------------------------------
print('Indexing word vectors')

word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
print('Found %s word vectors of word2vec' % len(word2vec.vocab))


#
# Process texts in datasets
# ----------------------------------------------------------------------------
print('Processing text dataset')


#
# The function "text_to_wordlist" is from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
# ----------------------------------------------------------------------------
def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.

    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    text = " ".join(text)

    # Clean the text
    text = preprocessing.word_patterns_replace(text)

    # Optionally, shorten words to their stems
    # Ex. >>> print(stemmer.stem("running"))
    #     run
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return text


#
# Train non-static-word2vec
# The function "train_word2vec" is from
# https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras/blob/master/w2v.py
# ----------------------------------------------------------------------------
def train_word2vec(sentences, word_index, num_features=300, min_word_count=1, context=10):
    """
        Trains, saves, loads Word2Vec model
        Returns initial weights for embedding layer.

        inputs:
        sentences       # list [sentence1, sentence2, ...]
        word_index      # tokenizer.word_index
        num_features    # Word vector dimensionality
        min_word_count  # Minimum word count
        context         # Context window size
    """
    from os.path import join, exists, split
    from gensim.models import word2vec

    model_dir = 'word2vec_models'
    model_name = "{:d}features_{:d}minwords_{:d}context".format(num_features, min_word_count, context)
    model_name = join(model_dir, model_name)
    if exists(model_name):
        embedding_model = word2vec.Word2Vec.load(model_name)
        print('Loading existing Word2Vec model \'%s\'' % split(model_name)[-1])
    else:
        # Set values for various parameters
        num_workers = 2  # Number of threads to run in parallel
        downsampling = 1e-3  # Downsample setting for frequent words
        print("Training Word2Vec model...")
        embedding_model = word2vec.Word2Vec(sentences, workers=num_workers,
                                            size=num_features, min_count=min_word_count,
                                            window=context, sample=downsampling)

        # If we don't plan to train the model any further, calling
        # init_sims will make the model much more memory-efficient.
        embedding_model.init_sims(replace=True)

        # Saving the model for later use. You can load it later using Word2Vec.load()
        if not exists(model_dir):
            os.mkdir(model_dir)
        print('Saving Word2Vec model \'%s\'' % split(model_name)[-1])
        embedding_model.save(model_name)

    #  add unknown words
    nb_words = min(MAX_NB_WORDS, len(word_index)) + 1

    embedding_weights = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if word in embedding_model:
            embedding_weights[i] = embedding_model[word]
        else:
            embedding_weights[i] = np.random.uniform(-0.25, 0.25, embedding_model.vector_size)
    return embedding_weights


#
# Read Training data
# ----------------------------------------------------------------------------
train_texts_1 = []
train_texts_2 = []
train_labels = []
with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        train_texts_1.append(text_to_wordlist(values[3]))
        train_texts_2.append(text_to_wordlist(values[4]))
        train_labels.append(int(values[5]))
print('Found %s texts in train.csv' % len(train_texts_1))

#
# Read Validation data
# ----------------------------------------------------------------------------
valid_texts_1 = []
valid_texts_2 = []
valid_labels = []
with codecs.open(VALID_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        valid_texts_1.append(text_to_wordlist(values[3]))
        valid_texts_2.append(text_to_wordlist(values[4]))
        valid_labels.append(int(values[5]))
print('Found %s texts in validation.csv' % len(valid_texts_1))

#
# Read Testing data
# ----------------------------------------------------------------------------
test_texts_1 = []
test_texts_2 = []
test_ids = []
with codecs.open(TEST_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        test_texts_1.append(text_to_wordlist(values[1]))
        test_texts_2.append(text_to_wordlist(values[2]))
        test_ids.append(values[0])
print('Found %s texts in test.csv' % len(test_texts_1))


#
# Vectorize texts and turn texts into sequences
# (=list of word indexes, where the word of rank i in the dataset (starting at 1) has index i).
# Ex. ['i am pig', 'you are queen'] ===> [ [1, 2, 3], [4, 5, 6] ]
#     word_index = {'i':1, 'am':2, 'pig':3, 'you':4, 'are':5, 'queen':6}
# ----------------------------------------------------------------------------
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(train_texts_1 + train_texts_2 + valid_texts_1 + valid_texts_2 + test_texts_1 + test_texts_2)

#
# Tokenize training data
# ----------------------------------------------------------------------------
train_sequences_1 = tokenizer.texts_to_sequences(train_texts_1)
train_sequences_2 = tokenizer.texts_to_sequences(train_texts_2)
train_data_1 = pad_sequences(train_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
train_data_2 = pad_sequences(train_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
train_labels = np.array(train_labels)
print('Shape of data tensor:', train_data_1.shape)
print('Shape of label tensor:', train_labels.shape)

#
# Tokenize validation data
# ----------------------------------------------------------------------------
valid_sequences_1 = tokenizer.texts_to_sequences(valid_texts_1)
valid_sequences_2 = tokenizer.texts_to_sequences(valid_texts_2)
valid_data_1 = pad_sequences(valid_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
valid_data_2 = pad_sequences(valid_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
valid_labels = np.array(valid_labels)
print('Shape of data tensor:', valid_data_1.shape)
print('Shape of label tensor:', valid_labels.shape)

#
# Tokenize testing data
# ----------------------------------------------------------------------------
test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)
test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
test_ids = np.array(test_ids)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))


#
# Prepare embeddings: construct a matrix  for mapping (word_id ==> word_vec)
# The embedding matrix with (number_of_words, embedding_dimension)
# ----------------------------------------------------------------------------
print('Preparing embedding matrix')

nb_words = min(MAX_NB_WORDS, len(word_index)) + 1

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

embedding_matrix_non_static = train_word2vec(train_texts_1 + train_texts_2 + valid_texts_1 + valid_texts_2 + test_texts_1 + test_texts_2, word_index)


#
# Sample predefined train/validation data
# ----------------------------------------------------------------------------
data_1_train = np.vstack((train_data_1, train_data_2))
data_2_train = np.vstack((train_data_2, train_data_1))
labels_train = np.concatenate((train_labels, train_labels))

data_1_valid = np.vstack((valid_data_1, valid_data_2))
data_2_valid = np.vstack((valid_data_2, valid_data_1))
labels_valid = np.concatenate((valid_labels, valid_labels))


weight_val = np.ones(len(labels_valid))
if re_weight:
    weight_val *= 0.472001959
    weight_val[labels_valid == 0] = 1.309028344


#
# Define the model structure
# ----------------------------------------------------------------------------
embedding_layer = Embedding(nb_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
embedding_layer_ns = Embedding(nb_words,
                               EMBEDDING_DIM,
                               weights=[embedding_matrix_non_static],
                               input_length=MAX_SEQUENCE_LENGTH,
                               trainable=False)

lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)


sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

#
# Static word2vec embedded
# ----------------------------------------------------------------------------
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1 = lstm_layer(embedded_sequences_1)

embedded_sequences_2 = embedding_layer(sequence_2_input)
y1 = lstm_layer(embedded_sequences_2)

#
# Non-Static word2vec embedded
# ----------------------------------------------------------------------------
embedded_sequences_3 = embedding_layer_ns(sequence_1_input)
d1 = Dropout(rate_drop_dense)(embedded_sequences_3)
x2 = lstm_layer(d1)

embedded_sequences_4 = embedding_layer_ns(sequence_2_input)
d2 = Dropout(rate_drop_dense)(embedded_sequences_4)
y2 = lstm_layer(d2)

#
# Merge all embedded sequences
# ----------------------------------------------------------------------------
merged = concatenate([x1, y1, x2, y2])
merged = Dropout(rate_drop_dense)(merged)

merged = BatchNormalization()(merged)
merged = Dense(num_dense, activation=act)(merged)
merged = Dropout(rate_drop_dense)(merged)

merged = BatchNormalization()(merged)
preds = Dense(1, activation='sigmoid')(merged)


#
# Add class weight
# ----------------------------------------------------------------------------
if re_weight:
    class_weight = {0: 1.309028344, 1: 0.472001959}
else:
    class_weight = None

#
# Train the model
# ----------------------------------------------------------------------------
model = Model(inputs=[sequence_1_input, sequence_2_input],
              outputs=preds)
model.compile(loss='binary_crossentropy',
              optimizer='nadam',
              metrics=['acc'])
# model.summary()
print(STAMP)

early_stopping = EarlyStopping(monitor='val_loss', patience=3)
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

hist = model.fit([data_1_train, data_2_train],
                 labels_train,
                 validation_data=([data_1_valid, data_2_valid], labels_valid, weight_val),
                 epochs=200, batch_size=2048, shuffle=True,
                 class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])

model.load_weights(bst_model_path)
bst_val_score = min(hist.history['val_loss'])


#
# Make the submission
# ----------------------------------------------------------------------------
print('Start making the submission before fine-tuning')

preds = model.predict([test_data_1, test_data_2], batch_size=8192, verbose=1)
preds += model.predict([test_data_2, test_data_1], batch_size=8192, verbose=1)
preds /= 2

submission = pd.DataFrame({'test_id': test_ids, 'is_duplicate': preds.ravel()})
submission.to_csv('%.4f_' % (bst_val_score) + STAMP + '.csv', index=False)