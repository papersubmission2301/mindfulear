import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report as cl

import tensorflow as tf
import keras
from keras import models

#load_data
X_test_co = np.load('X_test_co_label_mfcc.npy')
y_test_co = np.load('y_test_co_label_mfcc.npy')

X_test_cl = np.load('X_test_cl_label_mfcc.npy')
y_test_cl = np.load('y_test_cl_label_mfcc.npy')

X_test_eq = np.load('X_test_eq_label_mfcc.npy')
y_test_eq = np.load('y_test_eq_label_mfcc.npy')

#load_models
model_filepath_co = 'model/mindfulness_skills_improvement/concentration/'
model_filepath_cl = 'model/mindfulness_skills_improvement/clarity/'
model_filepath_eq = 'model/mindfulness_skills_improvement/equanimity/'
sv_model_co = models.load_model(model_filepath_co)
sv_model_cl = models.load_model(model_filepath_cl)
sv_model_eq = models.load_model(model_filepath_eq)

#prediction
y_pred_co = sv_model_co.predict(X_test_co)
y_pred_co = np.argmax(y_pred_co, axis = 1)
print(cl(y_pred_co, y_test_co))

y_pred_cl = sv_model_cl.predict(X_test_cl)
y_pred_cl = np.argmax(y_pred_cl, axis = 1)
print(cl(y_pred_cl, y_test_cl))

y_pred_eq = sv_model_eq.predict(X_test_eq)
y_pred_eq = np.argmax(y_pred_eq, axis = 1)
print(cl(y_pred_eq, y_test_eq))

