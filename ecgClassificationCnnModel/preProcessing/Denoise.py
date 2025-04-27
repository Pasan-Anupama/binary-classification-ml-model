#This code contains the denoising of ECG signals

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

def bandpass_filter(signal, fs, lowcut=0.5, highcut=40):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, signal)

def notch_filter(signal, fs, freq=50, Q=30):
    nyq = 0.5 * fs
    freq = freq / nyq
    b, a = iirnotch(freq, Q)
    return filtfilt(b, a, signal)

