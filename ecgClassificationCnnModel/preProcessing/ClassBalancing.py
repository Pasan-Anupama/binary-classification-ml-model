# Contains the code for class balancing

from imblearn.over_sampling import  SMOTE

def balance_calsses(X, y):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X.reshape(X.shape[0], -1), y)