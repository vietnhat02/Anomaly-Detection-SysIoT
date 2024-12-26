from sklearn.metrics import (roc_curve, roc_auc_score, precision_score, 
                           recall_score, accuracy_score, f1_score, matthews_corrcoef)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def find_best_threshold(anomaly_scores, true_labels):
    fpr, tpr, thresholds = roc_curve(true_labels, anomaly_scores)
    auc_score = roc_auc_score(true_labels, anomaly_scores)
    gini = 2 * auc_score - 1

    best_threshold = thresholds[np.argmax(tpr)]
    y_pred = (anomaly_scores >= best_threshold).astype(int)

    metrics = [{
        'threshold': best_threshold,
        'f1': f1_score(true_labels, y_pred, average='weighted', zero_division=0),
        'precision': precision_score(true_labels, y_pred, zero_division=0),
        'recall': recall_score(true_labels, y_pred),
        'accuracy': accuracy_score(true_labels, y_pred),
        'mcc': matthews_corrcoef(true_labels, y_pred)
    }]

    return pd.DataFrame(metrics), fpr, tpr, best_threshold, auc_score, gini