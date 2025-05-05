# This code contains the main pipeline (try python -m models.Run to run)

from preProcessing.Denoise import bandpass_filter, notch_filter, remove_baseline
from preProcessing.Segment import extract_heartbeats
from preProcessing.ClassBalancing import balance_classes
from preProcessing.Normalization import normalize_beats
from preProcessing.Load import load_ecg
from preProcessing.Labels import create_labels
from models.TrainOneRecord import train_model
from models.Evaluate import evaluate_model
from models.Evaluate import plot_metrics, evaluate_model

def process_record(record_id, data_dir):
    # 1. Load signal with annotations
    signal, eventSamples, fs, ann = load_ecg(record_id, data_dir)
    print("Total events in ", record_id, " record : ", len(eventSamples))
    print(f"Total annotations: {len(ann.sample)}")
    
    # 2. Denoising
    signal = bandpass_filter(signal, fs)
    signal = notch_filter(signal, fs)
    signal = remove_baseline(signal, fs)
    
    # 3. Heartbeat extraction (beats -> segments, valid_rpeaks -> R peaks samples)
    beats, valid_rpeaks = extract_heartbeats(signal, fs, ann.sample)
    print(f"Extracted {len(beats)} valid beats")
    
    # 4. Normalization
    beats = normalize_beats(beats)  
    
    # 5. Create labels
    labels = create_labels(valid_rpeaks, ann)
    print("Number of labels generated : ", len(labels))
    
    # 6. Class balancing
    beats_flat = beats.reshape(beats.shape[0], -1)
    X_balanced, y_balanced = balance_classes(beats_flat, labels)
    X_balanced = X_balanced.reshape(-1, beats.shape[1], 1)
    print("Balanced segments : ", len(X_balanced))
    print("Balanced annotations : ", len(y_balanced))

    # Return X_balanced -> class balanced beats(segments), y_balanced -> class balanced R-peak annotations(1 for Abnormal and 
    # 0 for Normal)
    return X_balanced, y_balanced

if __name__ == "__main__":
    X, y = process_record('101', 'data/mitdb')
    model, history = train_model(X, y)
    plot_metrics(history)
    evaluate_model(model, X, y)