import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import scipy as sp
from scipy.signal import butter, filtfilt, medfilt, lfilter, sosfilt

np.random.seed(42)
def find_label(dataframe, target_column, PID, session):
    iteration = len(dataframe['PID'])
    for i in range(iteration):
        if int(dataframe['PID'][i]) == int(PID) and int(dataframe['Session'][i]) == int(session):
            value = dataframe[target_column][i]
            break
        else:
            value = 0
    return value

def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    import numpy as np
    from math import factorial
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def data_reshape(data):
    x = data
    y = x.reshape(x.shape[0], -1, 1)
    return y

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


scaler = MinMaxScaler(feature_range=(0, 1))
def processing(train, test):
    train = train.reshape(train.shape[0], -1)
    test = test.reshape(test.shape[0], -1)
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)
    return train, test



path = ""
test_session = [3]

positive_data_concentration_train = []
positive_data_clarity_train = []
positive_data_equanimity_train = []

negative_data_concentration_train = []
negative_data_clarity_train = []
negative_data_equanimity_train = []

positive_data_concentration_test = []
positive_data_clarity_test = []
positive_data_equanimity_test = []

negative_data_concentration_test = []
negative_data_clarity_test = []
negative_data_equanimity_test = []

for file1 in glob.glob(path +"*.wav"):
    PID = int(file1[44:47])
    try:
        session = int(file1[49:51])
    except:
        session = int(file1[49])
    
    if file1[-17:-4] == 'technic_audio':
        data, fs = librosa.load(file1, sr=11025)
        c_value_post = find_label(y_post, 'Concentration_1', PID, session)
        s_value_post = find_label(y_post, 'SensoryClarity_1', PID, session)
        e_value_post = find_label(y_post, 'Equanimity_1', PID, session)

        c_value_pre = find_label(y_pre, 'Concentration_1', PID, session)
        s_value_pre = find_label(y_pre, 'SensoryClarity_1', PID, session)
        e_value_pre = find_label(y_pre, 'Equanimity_1', PID, session)
        
        if int(session) in test_session:
            for i in range(0, len(data), int(fs*120)):
                data_chunk = data[i:int(i+fs*120)]
                if i+int(fs*120) >len(data):
                    continue
                data_chunk = librosa.feature.mfcc(y = data_chunk, sr=fs, n_mfcc=40, n_fft = 512, win_length=512, hop_length = 256)
                
                if c_value_post > c_value_pre:
                    positive_data_concentration_test.append(data_chunk)
                else:
                    negative_data_concentration_test.append(data_chunk)

                if s_value_post > s_value_pre:
                    positive_data_clarity_test.append(data_chunk)
                else:
                    negative_data_clarity_test.append(data_chunk)

                if e_value_post > e_value_pre:
                    positive_data_equanimity_test.append(data_chunk)
                else:
                    negative_data_equanimity_test.append(data_chunk)
        else:
            for i in range(0, len(data), int(fs*120)):
                data_chunk = data[i:int(i+fs*120)]
                if i+int(fs*120) >len(data):
                    continue
                data_chunk = librosa.feature.mfcc(y = data_chunk, sr=fs, n_mfcc=40, n_fft = 512, win_length=512, hop_length = 256)

                if c_value_post > c_value_pre:
                    positive_data_concentration_train.append(data_chunk)
                else:
                    negative_data_concentration_train.append(data_chunk)

                if s_value_post > s_value_pre:
                    positive_data_clarity_train.append(data_chunk)
                else:
                    negative_data_clarity_train.append(data_chunk)

                if e_value_post > e_value_pre:
                    positive_data_equanimity_train.append(data_chunk)
                else:
                    negative_data_equanimity_train.append(data_chunk)

positive_data_concentration_train = np.array(positive_data_concentration_train)
negative_data_concentration_train = np.array(negative_data_concentration_train)

positive_data_concentration_test = np.array(positive_data_concentration_test)
negative_data_concentration_test = np.array(negative_data_concentration_test)

positive_data_clarity_train = np.array(positive_data_clarity_train)
negative_data_clarity_train = np.array(negative_data_clarity_train)

positive_data_clarity_test = np.array(positive_data_clarity_test)
negative_data_clarity_test = np.array(negative_data_clarity_test)

positive_data_equanimity_train = np.array(positive_data_equanimity_train)
negative_data_equanimity_train = np.array(negative_data_equanimity_train)

positive_data_equanimity_test = np.array(positive_data_equanimity_test)
negative_data_equanimity_test = np.array(negative_data_equanimity_test)

X_train_pos_co = data_reshape(positive_data_concentration_train)
X_train_neg_co = data_reshape(negative_data_concentration_train)

X_test_pos_co = data_reshape(positive_data_concentration_test)
X_test_neg_co = data_reshape(negative_data_concentration_test)

X_train_pos_cl = data_reshape(positive_data_clarity_train)
X_train_neg_cl = data_reshape(negative_data_clarity_train)

X_test_pos_cl = data_reshape(positive_data_clarity_test)
X_test_neg_cl = data_reshape(negative_data_clarity_test)

X_train_pos_eq = data_reshape(positive_data_equanimity_train)
X_train_neg_eq = data_reshape(negative_data_equanimity_train)

X_test_pos_eq = data_reshape(positive_data_equanimity_test)
X_test_neg_eq = data_reshape(negative_data_equanimity_test)

y_train_co = np.concatenate((np.ones((len(X_train_pos_co))), np.zeros((len(X_train_neg_co)))))
y_train_cl = np.concatenate((np.ones((len(X_train_pos_cl))), np.zeros((len(X_train_neg_cl)))))
y_train_eq = np.concatenate((np.ones((len(X_train_pos_eq))), np.zeros((len(X_train_neg_eq)))))

y_test_co = np.concatenate((np.ones((len(X_test_pos_co))), np.zeros((len(X_test_neg_co)))))
y_test_cl = np.concatenate((np.ones((len(X_test_pos_cl))), np.zeros((len(X_test_neg_cl)))))
y_test_eq = np.concatenate((np.ones((len(X_test_pos_eq))), np.zeros((len(X_test_neg_eq)))))

X_train_co, y_train_co = unison_shuffled_copies(X_train_co, y_train_co)
X_train_cl, y_train_cl = unison_shuffled_copies(X_train_cl, y_train_cl)
X_train_eq, y_train_eq = unison_shuffled_copies(X_train_eq, y_train_eq)

X_test_co, y_test_co = unison_shuffled_copies(X_test_co, y_test_co)
X_test_cl, y_test_cl = unison_shuffled_copies(X_test_cl, y_test_cl)
X_test_eq, y_test_eq = unison_shuffled_copies(X_test_eq, y_test_eq)

X_train_co, X_test_co = processing(X_train_co, X_test_co)
X_train_cl, X_test_cl = processing(X_train_cl, X_test_cl)
X_train_eq, X_test_eq = processing(X_train_eq, X_test_eq)

np.save('X_train_co_label_mfcc.npy', X_train_co)
np.save('X_test_co_label_mfcc.npy', X_test_co)

np.save('X_train_cl_label_mfcc.npy', X_train_cl)
np.save('X_test_cl_label_mfcc.npy', X_test_cl)

np.save('X_train_eq_label_mfcc.npy', X_train_eq)
np.save('X_test_eq_label_mfcc.npy', X_test_eq)

np.save('y_train_co_label_mfcc.npy', y_train_co)
np.save('y_test_co_label_mfcc.npy', y_test_co)

np.save('y_train_cl_label_mfcc.npy', y_train_cl)
np.save('y_test_cl_label_mfcc.npy', y_test_cl)

np.save('y_train_eq_label_mfcc.npy', y_train_eq)
np.save('y_test_eq_label_mfcc.npy', y_test_eq)

