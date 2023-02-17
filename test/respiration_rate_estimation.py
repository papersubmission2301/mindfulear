import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error

X_test = np.load('audio_br_test_data_mfcc.npy')
y_test = np.load('audio_br_test_label_mfcc.npy')


model_filepath = 'model/respiration_rate_estimation/'
sv_model = tf.keras.models.load_model(model_filepath)
y_pred = sv_model.predict(X_test)

print(mean_absolute_error(y_pred, y_test))


