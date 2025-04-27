# This code contains the code for Feature extraction

from scipy.signal import find_peaks
import numpy as np

def extract_waveform_features(beat):
    """Extract time-domain features"""
    peaks, _ = find_peaks(beat, height=0.5)
    troughs, _ = find_peaks(-beat, height=0.5)
    return {
        'num_peaks' : len(peaks),
        'amplitude': np.max(beat) - np.min(beat),
        'r_peak_height': beat[peaks[0]] if len(peaks) > 0 else 0
    }
    