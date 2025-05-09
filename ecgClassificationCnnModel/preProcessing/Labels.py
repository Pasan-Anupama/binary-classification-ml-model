# This code is to assign 0 for normal beats (N, L, R) and assign 1 for abnormal beats. Then return the array of labels. 

import numpy as np

# The function gets rpeaks by neurokit library and annotations of the ECG by atr files in db
def create_labels(rpeaks, annotation):
    """Create labels only for valid extracted beats"""
    labels = []
    beat_symbols = annotation.symbol
    annotation_samples = annotation.sample  # Eg : annotation_samples = [100, 350, 600, 900] -> Annotation sample contains the
                                            # locations of the events/annotation samples. 
    
    for peak in rpeaks:
        idx = np.argmin(np.abs(annotation_samples - peak))
        labels.append(0 if beat_symbols[idx] in ['N','L','R'] else 1)
    
    return np.array(labels)

# Calculation : 
# annotation_samples - peak = [100-355, 350-355, 600-355, 900-355] = [-255, -5, 245, 545]
# np.abs(...) = [255, 5, 245,545]
# np.argmin(...) returns 1 (since 5 is the smallest difference)
# So, idx[1] is the 350 value.
# Then get the symbol relevant to 350 by beat_symbols and assign 1 or 0. (0 -> Normal, 1 -> Abnormal)
