import logging
import sys

if 'stdout' not in globals():
    stdout = sys.stdout
if 'stderr' not in globals():
    stderr = sys.stderr


class StdOutLogger(object):
    def __init__(self, orig, writer):
        self.orig = orig
        self.writer = writer
        self.buffer = ''

    def write(self, s):
        if s.endswith('\n'):
            self.writer(self.buffer + s[0:-1])
            self.buffer = ''
        else:
            self.buffer = self.buffer + s

    def flush(self):
        pass

    def __getattr__(self, name):
        return object.__getattribute__(self.orig, name)


def init_logger(output_file='output.log',
                log_format='%(asctime)-15s [%(levelname)s] %(message)s'):
    logger = logging.getLogger()
    logger.handlers = []
    logger.addHandler(logging.StreamHandler(stdout))
    if output_file is not None:
        logger.addHandler(logging.FileHandler(output_file))
    for x in logger.handlers:
        x.setFormatter(logging.Formatter(log_format))
    logger.setLevel(logging.INFO)
    sys.stdout = StdOutLogger(stdout, lambda s: logger.info(s))
    sys.stderr = StdOutLogger(stderr, lambda s: logger.error(s))
    return logger


logger = init_logger('../log/clear-cnn1.log')

#import gzip
#import json
import pandas as pd
import numpy as np
#from gensim.models.fasttext import FastText

import tensorflow as tf
from keras.models import load_model
from keras import backend as K

tf.set_random_seed(99)

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
K.set_session(session)






def load_data_indices(filename, salary_type='annually'):

    min_salary, max_salary, base_y = get_salary_baseline(salary_type)

    indices = []
    counter = 0
    with gzip.open(filename.replace('.txt.gz', '.ndjson.gz'), 'rt') as f:
        for line in f:
            counter += 1
            if counter % 1000000 == 0:
                print(f'Load {counter} documents')
            if len(line) == 0:
                break
            job = json.loads(line)
            salary_dict = job.get('salary')
            if salary_dict is None or salary_dict.get('type') != salary_type:
                continue
            target = salary_dict.get('min')
            if target < min_salary or target > max_salary:
                continue
            indices.append(counter)

    return np.array(indices)


indices_all = load_data_indices(job_filename, salary_type='annually')

# indices_subset = indices_all
np.random.shuffle(indices_all)
indices_subset = indices_all[0:400000]
len(indices_subset)

from sklearn.model_selection import train_test_split
indices_train, indices_test = train_test_split(indices_subset, test_size=0.1, random_state=1)
print(f'indices_train: {len(indices_train)}')
print(f'indicies_test: {len(indices_test)}')

indices_subtrain, indices_valid = train_test_split(indices_train, test_size=0.1, random_state=1)
X_train, y_train = load_data(job_filename, indices=indices_subtrain)
X_valid, y_valid = load_data(job_filename, indices=indices_valid)
print(f'indices_subtrain: {len(indices_subtrain)}')
print(f'indicies_valid: {len(indices_valid)}')

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding, LSTM
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, SpatialDropout1D
from keras.layers import Conv2D, MaxPool2D, Reshape, BatchNormalization, Lambda, Add, Reshape, Permute
from keras.layers import RepeatVector, Multiply, Activation
from keras.layers.merge import Concatenate
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.models import load_model
from keras.optimizers import Adam

import h5py


K.clear_session()

max_len=22+410+43+36 #11
embed_size=200
inputs = Input(shape=(max_len,))

weights = model_gensim.wv.vectors
embedding = Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1],
                      weights=[weights], trainable=False)

x_o = Lambda(lambda x: x[:,22+410+43+36:])(inputs)
x = embedding(Lambda(lambda x: x[:,:22+410+43+36])(inputs))

SINGLE_ATTENTION_VECTOR = True

def attention_3d_block(inputs, ts):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    TIME_STEPS = ts
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

def create_crnn_model(x, ts, units=100):
    x_f = GRU(units, return_sequences=True)(x)
    x_f = attention_3d_block(x_f, ts)
    x_f = Flatten()(x_f)
    x_b = GRU(units, return_sequences=True, go_backwards=True)(x)
    x_b = attention_3d_block(x_b, ts)
    x_b = Flatten()(x_b)

    conv_blocks = []
    for sz in [1,2,3]:
        conv = Convolution1D(filters=30,
                             kernel_size=sz,
                             padding="valid",
                             strides=1)(x)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        conv = MaxPooling1D(pool_size=int(ts/5),
                            strides=int(ts/10))(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)
    conv = Concatenate()(conv_blocks)
    conv = Dropout(0.2)(conv)
    conv = Dense(units, activation='relu')(conv)
    conv = BatchNormalization()(conv)

    return Concatenate()([x_f, x_b, conv])

def create_res_model(x, x_b, units=300):
    # https://arxiv.org/pdf/1603.05027.pdf
    x_res = BatchNormalization()(x)
    if x_b is not None:
        x_res = Concatenate()([x_res, x_b])
    x_res = Dropout(0.2)(x_res)
    x_res = Dense(units, activation='relu')(x_res)
    x_res = BatchNormalization()(x_res)
    x_res = Dropout(0.2)(x_res)
    x_res = Dense(units, activation='relu')(x_res)
    return Add()([x, x_res])

x_t = create_crnn_model(Lambda(lambda x: x[:,         :22,:])(x), 22)
x_c = create_crnn_model(Lambda(lambda x: x[:,22       :22+410,:])(x), 410)
x_r = create_crnn_model(Lambda(lambda x: x[:,22+410   :22+410+43,:])(x), 43)
x_w = create_crnn_model(Lambda(lambda x: x[:,22+410+43:22+410+43+36,:])(x), 36)

x = Concatenate()([x_t, x_o])
x = Dropout(0.2)(x)
x = Dense(300, activation='relu')(x)

x = create_res_model(x, x_r)
x = create_res_model(x, x_w)
x = create_res_model(x, x_c)

x = Dropout(0.2)(x)
x = Dense(200, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Dense(100, activation='relu')(x)
x = BatchNormalization()(x)

x = Dropout(0.2)(x)
x = Dense(1, activation='relu')(x)
model = Model(inputs, x)

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

model_path = 'model/clear-salary-cnn1.hdf5'

model.summary()

#print(X_train.shape, X_valid.shape)
callbacks = [EarlyStopping(monitor='val_loss',
                           patience=30,
                           verbose=0),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.8,
                               patience=5,
                               verbose=1,
                               #min_delta=1e-4,
                               mode='min'),
             ModelCheckpoint(filepath=model_path,
                             monitor='val_loss',
                             save_best_only=True,
                             mode='min')]

model.fit(X_train, y_train, epochs=200, batch_size=200,
          validation_data=(X_valid, y_valid),
          verbose=2, callbacks=callbacks, shuffle=False)

best_model = load_model(model_path)
best_model

from sklearn.metrics import mean_absolute_error

for upper_salary in [15000000, 10000000, 8000000, 6000000]:
    X_test, y_test = load_data(job_filename, max_doc=10000000, indices=indices_test, upper=upper_salary)
    y_pred = best_model.predict(X_test, batch_size=200)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'{mae} (<{upper_salary})')
