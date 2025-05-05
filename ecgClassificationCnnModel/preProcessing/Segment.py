# Contains the code for segmenting the ECG signal 

# NeuroKit2 is a Python toolbox for neurophysiological signal 
# processing (such as ECG, EDA, EMG, PPG, and more). It provides functions for: -> ECG processing (R-peak detection, 
# HRV analysis) -> EDA (electrodermal activity) analysis, -> Respiration signal processing

import neurokit2 as nk
import numpy as np

def extract_heartbeats(signal, fs, annotation_rpeaks=None, before=0.25, after=0.4, fixed_length=250):
    """
    Extract fixed-length heartbeats centered at R-peaks
    
    Args:
        signal: ECG signal
        fs: Sampling frequency (Hz)
        annotation_rpeaks: Optional pre-annotated R-peaks
        before: Seconds before R-peak (default 0.25)
        after: Seconds after R-peak (default 0.4)
        fixed_length: Target samples per beat (default 250)
        
    Returns:
        beats: Array of fixed-length beats (n_beats, fixed_length)
        valid_rpeaks: Array of used R-peak positions
    """
    # Clean signal and detect R-peaks
    cleaned = nk.ecg_clean(signal, sampling_rate=fs)
    rpeaks = annotation_rpeaks if annotation_rpeaks is not None else \
             nk.ecg_findpeaks(cleaned, sampling_rate=fs)['ECG_R_Peaks']
    
    beats = []
    valid_rpeaks = []
    
    for peak in rpeaks:
        # Calculate fixed window around R-peak
        center = int(peak)
        start = center - fixed_length//2
        end = center + fixed_length//2
        
        # Handle edge cases
        if start < 0:
            start, end = 0, fixed_length
        elif end > len(signal):
            start, end = len(signal)-fixed_length, len(signal)
        
        beat = signal[start:end]
        
        # Ensure exact length (in case of sampling rate rounding)
        if len(beat) < fixed_length:
            beat = np.pad(beat, (0, fixed_length - len(beat)))
        elif len(beat) > fixed_length:
            beat = beat[:fixed_length]
        
        beats.append(beat)
        valid_rpeaks.append(peak)
    
    # Here sample length has taken as 250 samples, because R peak lenght is about 0.694 seconds
    # Returns beats -> An array of fixed length heart beats (Segments), valid_rpeaks -> An array containing the sample position 
    # of R peaks. Eg : valid_rpeaks = [10, 234, 565] 
    return np.array(beats), np.array(valid_rpeaks)