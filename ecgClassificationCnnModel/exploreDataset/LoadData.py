# Function to load records in the MIT-BIH dataset and explore the leads, annotations and 
# other features

import wfdb

def interpret_data(record_id, data_dir):
    record = wfdb.rdrecord(f'{data_dir}/{record_id}')
    
    # .hea information
    print("=== HEADER (.hea) INFORMATION ===")
    
    recordNname = record.record_name
    print("Record Name : ", recordNname)
    
    samplingFreq = record.fs
    print("Sampling Frequency : ", samplingFreq)
    
    # Printing the channels in a record. MLII is Modifed Lead II which is commonly used and V5, V1 are periodiocal leads 
    signalNames = record.sig_name
    numberOfChannels = record.n_sig
    signalLength = record.sig_len
    adcGain = record.adc_gain
    baseLine = record.baseline
    print("Names of signal channels in ", record_id, "record are : ", signalNames)
    print("Number of channels : ", numberOfChannels)
    print("Signal Length : ", signalLength , " Samples")
    print("ADC Gain : ", adcGain)
    print("Signal Baseline : ", baseLine)
    
    # Printing the index of MLII, since it is the mostly used one in experiments. 
    # If it is at 0 -> channel 1 is used and if 1 -> channel 2 is used
    lead_ii_index = record.sig_name.index('MLII')
    print("Lead II (MLII) index : " , lead_ii_index)
    
    
    
    # Analysing the .dat file
    print("\n=== SIGNAL DATA (.dat) ===")
    
    signal = record.p_signal[:,lead_ii_index] # represent whole signal
    print("Signal length of Lead II : ", len(signal))
    print("Signal Shape : ", record.p_signal.shape)
    print("First five samples of channel 0 : ", record.p_signal[:5, 0])
    print("First five samples of channel 1 : ", record.p_signal[:5, 1])
    
    
    # Analysung .atr file
    print("\n=== ANNOTATION (.atr) INFORMATION ===")
    
    annotations = wfdb.rdann(f"{data_dir}/{record_id}", 'atr')
    numberOfAnnotations = len(annotations.sample)
    print("Number of annotations : ", numberOfAnnotations)
    print(f"First 10 annotation sample indices : {annotations.sample[:10]}") 
    print(f"First 10 annotation symbols : {annotations.symbol[:10]}")
    
def load_data(record_id, data_dir):
    record = wfdb.rdrecord(f'{data_dir}/{record_id}')
    ann = wfdb.rdann(f'{data_dir}/{record_id}', 'atr')
    return record.p_signal[:, 0], ann.sample, record.fs, ann
    
if __name__ == "__main__":
    interpret_data('100', 'data/mitdb')  