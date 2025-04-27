# This contains the code to test the denoising techniques implemented in Denoise.py 

import numpy as np
import matplotlib.pyplot as plt
from Denoise import bandpass_filter, notch_filter, remove_baseline
import wfdb

# Load real ECG data from MIT-BIH
record_id = '100'  # Test with record 100
record = wfdb.rdrecord(f'data/mitdb/{record_id}')
signal = record.p_signal[:, 0]  # Use lead II
fs = record.fs  # Get actual sampling frequency (360Hz)

# Apply denoise pipeline
bandpassed = bandpass_filter(signal, fs)
notchedFileterd = notch_filter(bandpassed, fs)
finalCLean = remove_baseline(notchedFileterd, fs)

# Create time axis in seconds
t = np.arange(len(signal)) / fs

# Plot results
plt.figure(figsize=(15, 12))

plt.subplot(4, 1, 1)
plt.plot(t, signal)
plt.title(f"Original MIT-BIH Record {record_id} (Lead II)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")

plt.subplot(4, 1, 2)
plt.plot(t, bandpassed)
plt.title("After Bandpass Filter (0.5-40 Hz)")

plt.subplot(4, 1, 3)
plt.plot(t, notchedFileterd)
plt.title("After Notch Filter (50 Hz Removal)")

plt.subplot(4, 1, 4)
plt.plot(t, finalCLean)
plt.title("After Baseline Removal")
plt.xlabel("Time (s)")

plt.tight_layout()
plt.savefig("ecg_denoising_results.png")
plt.show()

print(f"Denoising test completed on MIT-BIH record {record_id}")
print(f"Sampling rate: {fs} Hz, Signal length: {len(signal)/fs:.1f} seconds")