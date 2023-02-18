import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
   
import tensorflow as tf
import keras

from keras import Sequential
from keras import losses, models, optimizers
from keras.activations import relu, softmax
from keras.layers import Dense, Dropout, Input, Activation, LSTM

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

X_train = np.load('audio_br_train_data_mfcc.npy')
X_test = np.load('audio_br_test_data_mfcc.npy')

y_train = np.load('audio_br_train_label_mfcc.npy')
y_test = np.load('audio_br_test_label_mfcc.npy')

y_train= y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

#netwrok_architecture
inp = Input(shape=(X_train.shape[1:]))

x = LSTM(32, return_sequences=False, stateful=False, dropout = 0.5)(inp)
x = Dense(32)(x)
x = Dense(32)(x)
x = Activation("relu")(x)
out = Dense(1, activation = 'linear')(x)

model = models.Model(inputs=inp, outputs=out)

checkpoint_filepath_co = 'log_mfcc_v2/br/'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath_co,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=7)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(loss='MSE',
                optimizer=optimizer, metrics=[tf.keras.metrics.MeanAbsoluteError()])

# Hyper-parameters
BATCH_SIZE = 64
EPOCHS = 50

# Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
history = model.fit(X_train,
                    y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(X_test, y_test),
                    verbose=0, callbacks = [model_checkpoint_callback, early_stopping_callback])

