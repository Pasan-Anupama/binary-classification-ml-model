import wfdb
import neurokit2 as nk
import matplotlib.pyplot as plt
import numpy as np

def load_data(record_id, data_dir):
    record = wfdb.rdrecord(f'{data_dir}/{record_id}')
    ann = wfdb.rdann(f'{data_dir}/{record_id}', 'atr')
    return record.p_signal[:, 0], ann.sample, record.fs, ann

def plot_first_n_qrs(record_id, data_dir, n=3):
    ecg_signal, rpeaks_annotated, fs, ann = load_data(record_id, data_dir)
    print(f"Loaded record {record_id} with {len(ecg_signal)} samples at {fs} Hz")
    print(f"Number of annotated beats : {len(rpeaks_annotated)}")

    # Clean and detect R-peaks using NeuroKit2
    signals, info = nk.ecg_process(ecg_signal, sampling_rate=fs)
    # print("Informationm gained by ecg_process function : ", info)
    rpeaks = info['ECG_R_Peaks']
    print("No of R peaks : ", len(rpeaks))

    # Define window size around R peak -> Normally a QRS complex consist with 0.694 seconds, 
    # meaning it consit of 360*0.694 = 250 samples. Hence, 125 samples before and after a R peak is 
    # set to a window
    half_window_samples = 125

    # Create subplots: n rows, 1 column
    fig, axes = plt.subplots(n, 1, figsize=(8, 3 * n), sharex=False)
    
    # If n=1, axes is not an array, so make it iterable
    if n == 1:
        axes = [axes]

    for i in range(min(n, len(rpeaks))):
        r = rpeaks[i]
        start = max(r - half_window_samples, 0) # 125 samples before R peak
        end = min(r + half_window_samples, len(ecg_signal)) # 125 samples after R peak

        segment = ecg_signal[start:end]  # Collect 250 samples 
        time_axis = np.arange(start, end) / fs  

        ax = axes[i]
        ax.plot(time_axis, segment, label=f'QRS Complex {i+1}')
        ax.axvline(r / fs, color='red', linestyle='--', label='R peak')
        ax.set_title(f'Record {record_id} - QRS Complex {i+1}')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude (mV)')
        ax.legend()

    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    data_dir = 'data/mitdb' 
    record_id = '100'        
    plot_first_n_qrs(record_id, data_dir, n=3)
