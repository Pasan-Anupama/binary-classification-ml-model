import wfdb
import numpy as np
from sklearn.model_selection import train_test_split
import os

def load_mitbih_records(record_numbers, segment_length=200):
    X = []
    y = []

    for record_num in record_numbers:
        print(f"Processing record {record_num}...")
        record = wfdb.rdrecord(record_num, pn_dir='mitdb')
        annotation = wfdb.rdann(record_num, 'atr', pn_dir='mitdb')

        signal = record.p_signal[:, 0]  # MLII lead
        ann_samples = annotation.sample
        ann_symbols = annotation.symbol

        half_seg = segment_length // 2

        for idx, sample in enumerate(ann_samples):
            if sample - half_seg < 0 or sample + half_seg > len(signal):
                continue

            segment = signal[sample - half_seg : sample + half_seg]
            label = 0 if ann_symbols[idx] == 'N' else 1

            X.append(segment)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    print(f"Total segments: {len(X)}, Normal: {(y==0).sum()}, Abnormal: {(y==1).sum()}")
    return X, y

def main():
    record_numbers = [
        '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
        '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
        '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
        '209', '210', '212', '213', '214'
    ]

    X, y = load_mitbih_records(record_numbers, segment_length=200)

    # Normalize segments
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    os.makedirs('data', exist_ok=True)
    np.save('data/train_data.npy', X_train)
    np.save('data/train_labels.npy', y_train)
    np.save('data/test_data.npy', X_test)
    np.save('data/test_labels.npy', y_test)


    print("Saved train/test data and labels to 'data/' folder.")

if __name__ == "__main__":
    main()
