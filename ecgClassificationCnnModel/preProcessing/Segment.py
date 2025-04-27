# Contains the code for segmenting the ECG signal 

# NeuroKit2 is a Python toolbox for neurophysiological signal 
# processing (such as ECG, EDA, EMG, PPG, and more). It provides functions for: -> ECG processing (R-peak detection, HRV analysis)
# -> EDA (electrodermal activity) analysis, -> Respiration signal processing

import neurokit2 as nk
import numpy as np

def extract_heartbeats(signal, fs, before=0.25, after=0.4):
    """Extract fixed length segments centered at R-peaks"""
    cleaned = nk.ecg_clean(signal, sampling_rate=fs)
    rpeaks = nk.ecg_findpeaks(cleaned, sampling_rate=fs)['ECG_R_Peaks']
    
    segments = []
    for peak in rpeaks:
        start = int(peak - before*fs)
        end = int(peak + after*fs)
        if start >= 0 and end < len(signal):
            segment = signal[start:end]
            segments.append(segment)
        return np.array(segments), rpeaks