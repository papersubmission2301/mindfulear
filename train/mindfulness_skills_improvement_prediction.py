#import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import tensorflow as tf
import keras

from keras import Sequential
from keras import losses, models, optimizers
from keras.activations import relu, softmax
from keras.layers import Dense, Dropout, Input, Convolution2D, BatchNormalization, Activation, MaxPool2D, Flatten


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.set_visible_devices(gpus[1], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

#load_data
X_train_co = np.load('X_train_co_label_mfcc.npy')
X_test_co = np.load('X_test_co_label_mfcc.npy')

y_train_co = np.load('y_train_co_label_mfcc.npy')
y_test_co = np.load('y_test_co_label_mfcc.npy')

X_train_cl = np.load('X_train_cl_label_mfcc.npy')
X_test_cl = np.load('X_test_cl_label_mfcc.npy')

y_train_cl = np.load('y_train_cl_label_mfcc.npy')
y_test_cl = np.load('y_test_cl_label_mfcc.npy')

X_train_eq = np.load('X_train_eq_label_mfcc.npy')
X_test_eq = np.load('X_test_eq_label_mfcc.npy')

y_train_eq = np.load('y_train_eq_label_mfcc.npy')
y_test_eq = np.load('y_test_eq_label_mfcc.npy')


X_train_co = np.reshape(X_train_co,(X_train_co.shape[0],40, -1,1))
y_train1=keras.utils.to_categorical(y_train_co, num_classes=2, dtype='float32')

X_test_co = np.reshape(X_test_co,(X_test_co.shape[0],40, -1,1))
y_test1=keras.utils.to_categorical(y_test_co, num_classes=2, dtype='float32')

X_train_cl = np.reshape(X_train_cl,(X_train_cl.shape[0],40, -1,1))
y_train2=keras.utils.to_categorical(y_train_cl, num_classes=2, dtype='float32')

X_test_cl = np.reshape(X_test_cl,(X_test_cl.shape[0],40, -1,1))
y_test2=keras.utils.to_categorical(y_test_cl, num_classes=2, dtype='float32')

X_train_eq = np.reshape(X_train_eq,(X_train_eq.shape[0],40, -1,1))
y_train3=keras.utils.to_categorical(y_train_eq, num_classes=2, dtype='float32')

X_test_eq = np.reshape(X_test_eq,(X_test_eq.shape[0],40, -1,1))
y_test3=keras.utils.to_categorical(y_test_eq, num_classes=2, dtype='float32')

nclass = 2
#network architecture
inp = Input(shape=(X_train_co.shape[1:]))
x = Convolution2D(32, (3,3), padding="same")(inp)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = MaxPool2D()(x)

x = Convolution2D(32*2, (3,3), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = MaxPool2D()(x)

x = Convolution2D(32*3, (3,3), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = MaxPool2D()(x)

x = Convolution2D(32*3, (3,3), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = MaxPool2D()(x)

x = Flatten()(x)
x = Dense(64)(x)
x = Dense(64)(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
out = Dense(nclass, activation=softmax)(x)

model = models.Model(inputs=inp, outputs=out)


#checkpoint
checkpoint_filepath_cl = 'log_clarity/'
checkpoint_filepath_eq = 'log_equanimity/'
checkpoint_filepath_co = 'log_conc/'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath_eq,
    save_weights_only=False,
    monitor='val_precision',
    mode='max',
    save_best_only=True)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

#optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='BCE', optimizer=optimizer, metrics=['Precision'])

# Hyper-parameters
BATCH_SIZE = 8
EPOCHS = 30

# Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
history_co = model.fit(X_train_co,
                    y_train1,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(X_test_co, y_test1),
                    verbose=1, callbacks = [model_checkpoint_callback, early_stopping_callback])

history_cl = model.fit(X_train_cl,
                    y_train2,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(X_test_cl, y_test2),
                    verbose=1, callbacks = [model_checkpoint_callback, early_stopping_callback])

history_eq = model.fit(X_train_eq,
                    y_train3,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(X_test_eq, y_test3),
                    verbose=1, callbacks = [model_checkpoint_callback, early_stopping_callback])


