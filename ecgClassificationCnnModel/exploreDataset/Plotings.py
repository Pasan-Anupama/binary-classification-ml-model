import wfdb
import neurokit2 as nk 
import matplotlib.pyplot as plt 
import numpy as np 

def load_data(record_id, data_dir):
    record = wfdb.rdrecord(f'{data_dir}/{record_id}')
    ann = wfdb.rdann(f'{data_dir}/{record_id}','atr')
    
    return record.p_signal[:,0], ann.sample, record.fs, ann

def plot_first_two_qrs(record_id, data_dir):
    ecg_signal, rpeaks_annotated, fs, ann = load_data(record_id, data_dir)
    
    print(f"Loaded record {record_id} with {len(ecg_signal)} samples at {fs} Hz")
    print(f"Number of annotated beats : {len(rpeaks_annotated)}")
    
    # Use neurokit to clean and find R peaks
    signals, info = nk.ecg_process(ecg_signal, sampling_rate=fs)
    rpeaks = info['ECG_R_Peaks']
    
    # Delineate ECG waves (QRS onset and offset)
    delineate = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=fs, method='dwt')
    
    # Extract Q onset and S offset for first two QRS complexes
    q_onsets = delineate['ECG_Q_Onsets']
    s_offsets = delineate['ECG_S_Offsets']
     
    # Check if at least two QRS complexes are detected
    if len(q_onsets) < 2 or len(s_offsets) < 2:
        print("Less than two QRS complexes detected.")
        return
    
    for i in range(2):
        onset = q_onsets[i]
        offset = s_offsets[i]
        
        # Number of samples and duration in seconds
        num_samples = offset - onset
        duration_sec = num_samples / fs
        
        print(f"QRS Complex {i+1}:")
        print(f"  Start sample: {onset}")
        print(f"  End sample:   {offset}")
        print(f"  Number of samples: {num_samples}")
        print(f"  Duration (seconds): {duration_sec:.4f}")
        
        # Plot the QRS complex segment
        plt.figure(figsize=(6, 3))
        plt.plot(np.arange(onset, offset), ecg_signal[onset:offset], label=f'QRS Complex {i+1}')
        plt.title(f'Record {record_id} - QRS Complex {i+1}')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude (mV)')
        plt.legend()
        plt.show()
        
    # Plot full ECG with first two QRS complexes highlighted
    plt.figure(figsize=(15, 4))
    t = np.arange(len(ecg_signal)) / fs
    plt.plot(t, ecg_signal, label='ECG Signal')
    for i in range(2):
        onset = q_onsets[i]
        offset = s_offsets[i]
        plt.axvspan(onset/fs, offset/fs, color='red', alpha=0.3, label=f'QRS Complex {i+1}' if i == 0 else None)
    plt.title(f'Record {record_id} - ECG Signal with First Two QRS Complexes Highlighted')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude (mV)')
    plt.legend()
    plt.show()
    
    
if __name__ == "__main__":
    data_dir = 'data/mitdb'  
    record_id = '100'        
    plot_first_two_qrs(record_id, data_dir)