#This code contains the code for testing the implemented segmenting technique

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import matplotlib.pyplot as plt
from preProcessing.Segment import extract_heartbeats
import wfdb

# Load 100th sample with annotations
record_id = '101'
record = wfdb.rdrecord(f'data/mitdb/{record_id}')
annotation = wfdb.rdann(f'data/mitdb/{record_id}', 'atr')

signal = record.p_signal[:, 0] 
fs = record.fs
t = np.arange(len(signal)) / fs

# Extract heartbeats -> Segmenting
beats, rpeaks = extract_heartbeats(signal, fs)

# Get ground truth R-peaks from annotations
true_rpeaks = annotation.sample
true_labels = annotation.symbol

plt.figure(figsize=(15, 8))

# Plot full signal with detected vs true R-peaks
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.plot(rpeaks/fs, signal[rpeaks], 'rx', label='Detected R-peaks', markersize=8)
plt.plot(true_rpeaks/fs, signal[true_rpeaks], 'g+', label='True Annotations', markersize=5)
plt.title(f"MIT-BIH Record {record_id} - R-peak Detection")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.legend()

# Plot first 5 extracted beats with their labels
plt.subplot(2, 1, 2)
for i in range(min(5, len(beats))):
    beat_time = np.linspace(-0.25, 0.4, len(beats[i]))  # 250ms before to 400ms after R-peak
    plt.plot(beat_time, beats[i], label=f'Beat {i+1} ({true_labels[i]})')
plt.title("Extracted Heartbeat Segments")
plt.xlabel("Time relative to R-peak (s)")
plt.ylabel("Amplitude (mV)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("ecg_segmentation_results.png")
plt.show()

# Detection accuracy
detected_in_truth = np.sum(np.isin(rpeaks, true_rpeaks))
precision = detected_in_truth / len(rpeaks)
recall = detected_in_truth / len(true_rpeaks)


print(f"\nSegmentation Performance on Record {record_id}:")
print(f"- Detected beats: {len(beats)}")
print(f"- Ground truth beats: {len(true_rpeaks)}")
print(f"- Precision: {precision:.1%}")
print(f"- Recall: {recall:.1%}")
print(f"- Average beat duration: {len(beats[0])/fs*1000:.1f} ms")