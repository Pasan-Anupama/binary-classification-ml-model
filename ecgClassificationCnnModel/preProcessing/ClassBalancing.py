# Contains the code for class balancing
# Uses Synthetic Minority Over-sampling Technique(SMOTE) algorithm

from imblearn.over_sampling import SMOTE
import numpy as np

def balance_classes(X, y):
    # print(f"Input shapes - X: {X.shape}, y: {y.shape}")  # Debug
    
    # Ensure y is 1D
    y = np.ravel(y)
    
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X, y)