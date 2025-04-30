# This contains the code to test the denoising techniques implemented in Denoise.py 

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import matplotlib.pyplot as plt
from preProcessing.Denoise import bandpass_filter, notch_filter, remove_baseline
import wfdb
import neurokit2 as nk 

record_id = '100' 
record = wfdb.rdrecord(f'data/mitdb/{record_id}')
signal = record.p_signal[:, 0]  # Use lead II
fs = record.fs  # Sampling frequency (typically 360 Hz)

# Detect R-peaks using NeuroKit2 on raw signal or bandpassed signal -> Uses own algorithm rather than searchimg in 
# the annotations -> NeuroKit2 detects R-peaks based on the steepness of the absolute gradient of the ECG and finds 
# local maxima in QRS complexes
bandpassed_full = bandpass_filter(signal, fs)
rpeaks_dict = nk.ecg_findpeaks(bandpassed_full, sampling_rate=fs)
rpeaks = rpeaks_dict['ECG_R_Peaks']

# Select one R-peak to plot (e.g., the 10th beat)
beat_index = 10
rpeak_pos = rpeaks[beat_index]

# Define window around R-peak (e.g., 0.5 seconds before and after)
window_before = int(0.5 * fs)
window_after = int(0.5 * fs)

start = max(rpeak_pos - window_before, 0)
end = min(rpeak_pos + window_after, len(signal))

# Extract segments for each step
segment_raw = signal[start:end]
segment_bandpassed = bandpass_filter(segment_raw, fs)
segment_notched = notch_filter(segment_bandpassed, fs)
segment_baseline_removed = remove_baseline(segment_notched, fs)

# Time axis for the segment
t = np.arange(start, end) / fs

# Plot results for one beat
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(t, segment_raw)
plt.title(f"Original ECG Segment around R-peak #{beat_index} (Lead II)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")

plt.subplot(4, 1, 2)
plt.plot(t, segment_bandpassed)
plt.title("After Bandpass Filter (0.5-40 Hz)")

plt.subplot(4, 1, 3)
plt.plot(t, segment_notched)
plt.title("After Notch Filter (50 Hz Removal)")

plt.subplot(4, 1, 4)
plt.plot(t, segment_baseline_removed)
plt.title("After Baseline Removal")
plt.xlabel("Time (s)")

plt.tight_layout()
plt.savefig("ecg_denoising_one_beat.png")
plt.show()

print(f"Denoising test completed on one beat of MIT-BIH record {record_id}")
