#basic imports
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import librosa   

import scipy as sp
from scipy.signal import butter, filtfilt, medfilt, lfilter, sosfilt

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


#data_processing

path1 = './meditite/gpu_meditite/cohort_2_4/audio/'
path2 = './meditite/gpu_meditite/cohort_2_4/cc/'

test_id = [203]
train_data = []
train_label = []
test_data = []
test_label = []

for file1 in glob.glob(path1 +"*.wav"):
    PID1 = int(file1[44:47])
    try:
        session1 = int(file1[49:51])
        flag1 = 1
    except:
        session1 = int(file1[49])
        flag1 = 0
    if flag1 ==0:
        try:
            l1 = int(file1[52:54])
        except:
            l1 = int(file1[52])
    else:
        try:
            l1 = int(file1[53:55])
        except:
            l1 = int(file1[53])

        
    for file2 in glob.glob(path2 + "*.csv"):

        PID2 = int(file2[41:44])
        try:
            session2 = int(file2[46:48])
            flag2 = 1
        except:
            session2 = int(file2[46])
            flag2 = 0
        
        if flag2 ==0:
            try:
                l2 = int(file2[49:51])
            except:
                l2 = int(file2[49])
        else:
            try:
                l2 = int(file2[50:52])
            except:
                l2 = int(file2[50])
        if PID2 == PID1 and session2 == session1 and l2 == l1:
            if PID2 in test_id:
                if file1[-17:-4] == 'technic_audio':
                    data, fs = librosa.load(file1, sr=11025)
                    for i in range(0, len(data), int(fs*1)):
                        data_chunk = data[i:int(i+fs*15)]
                        if i+int(fs*15) >len(data):
                            continue
                        data_chunk = butter_bandpass_filter(data_chunk, 20, 1000, fs, order = 2)
                        data_chunk = savitzky_golay(data_chunk, 101, 3)
                        data_chunk = librosa.feature.mfcc(y = data_chunk, sr=fs, n_mfcc=40, n_fft = 512, win_length=512, hop_length = 256)
                        test_data.append(data_chunk)
                        
                df = pd.read_csv(file2, index_col = 0)
                label = df['breathing_rate'].values
                for i in range(len(label)):
                    if i+15 > len(label):
                        break
                    test_label.append(round(np.mean(label[i:i+15])))
            else:
                if file1[-17:-4] == 'technic_audio':
                    data, fs = librosa.load(file1, sr=11025)
                    data = butter_bandpass_filter(data, 20, 1000, fs, order = 2)
                    data = savitzky_golay(data, 101, 3)
                    for i in range(0, len(data), int(fs*1)):
                        data_chunk = data[i:int(i+fs*15)]
                        if i+int(fs*15) >len(data):
                            continue
                        data_chunk = butter_bandpass_filter(data_chunk, 20, 1000, fs, order = 2)
                        data_chunk = savitzky_golay(data_chunk, 101, 3)
                        data_chunk = librosa.feature.mfcc(y = data_chunk, sr=fs, n_mfcc=40, n_fft = 512, win_length= 512, hop_length = 256)
                        train_data.append(data_chunk)
                        
                df = pd.read_csv(file2, index_col = 0)
                label = df['breathing_rate'].values
                for i in range(len(label)):
                    if i+15 > len(label):
                        break
                    train_label.append(round(np.mean(label[i:i+15])))

#saving data
test_data = np.array(test_data)
np.save('audio_br_test_data_mfcc.npy', test_data)
del test_data
np.save('audio_br_train_label_mfcc.npy', train_label)
np.save('audio_br_test_label_mfcc.npy', test_label)

train_data = np.array(train_data)
np.save('audio_br_train_data_mfcc.npy', train_data)

