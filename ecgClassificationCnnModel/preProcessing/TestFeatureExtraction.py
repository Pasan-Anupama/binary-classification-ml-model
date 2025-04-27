# This code contains the code top test the Feature extraction 

import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from FeatureExtraction import extract_waveform_features
import wfdb 

def test_with_real_mitdb(record_id='100', beat_num=0):
    """
    Test feature extraction on real MIT-BIH data with annotation validation
    
    Args:
        record_id: MIT-BIH record number (e.g., '100')
        beat_num: Index of beat to test (default: first beat)
    """
    
    # 1. Load record and annotations
    record = wfdb.rdrecord(f'data/mitdb/{record_id}')
    ann = wfdb.rdann(f'data/mitdb/{record_id}', 'atr')
    
     # Get signal (Lead II) and sampling frequency
    fs = record.fs
    signal = record.p_signal[:, 0] 
    
     # 2. Extract one beat using annotation
    r_peak = ann.sample[beat_num]
    beat_type = ann.symbol[beat_num]  # Get physician's label (N, V, etc.)
    
    # Extract 250ms before to 400ms after R-peak (typical window)
    window_start = r_peak - int(0.25 * fs)
    window_end = r_peak + int(0.4 * fs)
    beat = signal[window_start:window_end]
    
     # 3. Extract and print features
    features = extract_waveform_features(beat)
    
    # 4. Visualization with ground truth
    t = np.linspace(-0.25, 0.4, len(beat))  # Time relative to R-peak
    
    plt.figure(figsize=(12, 5))
    plt.plot(t, beat, label=f'Beat {beat_num} ({beat_type})')
    
    # Mark R-peak position (ground truth at t=0)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='R-peak (t=0)')
    
    # Mark detected features
    peaks, _ = find_peaks(beat, height=0.5)
    troughs, _ = find_peaks(-beat, height=0.5)
    
    if len(peaks) > 0:
        plt.plot(t[peaks], beat[peaks], 'rx', markersize=10, label='Detected Peaks')
    if len(troughs) > 0:
        plt.plot(t[troughs], beat[troughs], 'go', markersize=8, label='Detected Troughs')
    
    plt.title(f"Real MIT-BIH Beat (Record {record_id}, {beat_type}-type)\n"
              f"Features: {features}")
    plt.xlabel("Time relative to R-peak (s)")
    plt.ylabel("Amplitude (mV)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 5. Quantitative validation
    print("\n=== Feature Validation ===")
    print(f"Beat type: {beat_type}")
    print(f"R-peak height: {features['r_peak_height']:.2f} mV")
    print(f"QRS amplitude: {features['amplitude']:.2f} mV")
    print(f"Detected peaks: {features['num_peaks']}")
    
    # Expected ranges for normal beats
    if beat_type == 'N':
        assert 0.5 < features['r_peak_height'] < 3.0, "Abnormal R-peak height"
        assert 0.8 < features['amplitude'] < 4.0, "Abnormal QRS amplitude"
        print("✅ Normal beat features within expected ranges")

if __name__ == "__main__":
    # Test with first normal beat from record 100
    test_with_real_mitdb(record_id='100', beat_num=0)
    
    # Test with a ventricular beat (if available)
    # test_with_real_mitdb(record_id='106', beat_num=10)  # Typically has V-beats