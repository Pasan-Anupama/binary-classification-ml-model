# This code contains the main pipeline

import numpy as np
from preProcessing.Denoise import bandpass_filter, notch_filter, remove_baseline
from preProcessing.Segment import extract_heartbeats
from preProcessing.ClassBalancing import balance_calsses
from preProcessing.Normalization import normalize_beats
from models.Train import train_model
from models.Evaluate import evaluate_model
from models.Evaluate import plot_metrics, evaluate_model

def process_record(record_id, data_dir):
    # 1. Load raw signal
    signal, _, fs = load_ecg(record_id, data_dir)
    
    # 2. Denoising
    signal = bandpass_filter(signal, fs)
    signal = notch_filter(signal, fs)
    signal = remove_baseline(signal, fs)
    
    # 3. Heartbeat extraction
    beats, rpeaks = extract_heartbeats(signal, fs)
    
    # 4. Normalization
    beats = normalize_beats(beats)
    
    # 5. Create labels (0=normal, 1=abnormal)
    labels = create_labels(rpeaks, fs)
    
    # 6. Class balancing
    beats_flat = beats.reshape(beats.shape[0], -1)
    X_balanced, y_balanced = balance_classes(beats_flat, labels)
    X_balanced = X_balanced.reshape(-1, beats.shape[1], 1)
    
    return X_balanced, y_balanced

if __name__ == "__main__":
    X, y = process_record('100', 'ecgClassification/data/mitdb')
    model, history = train_model(X, y)
    plot_metrics(history)
    evaluate_model(model, X, y)