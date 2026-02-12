import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score, precision_score, recall_score
from scipy.optimize import linear_sum_assignment

def clustering_accuracy(y_true, y_pred):
    
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)
    n_labels = max(len(labels_true), len(labels_pred))
    
    
    cost_matrix = np.zeros((n_labels, n_labels), dtype=np.int64)
    for i, label_true in enumerate(labels_true):
        for j, label_pred in enumerate(labels_pred):
            cost_matrix[i, j] = np.sum((y_true == label_true) & (y_pred == label_pred))
    
   
    row_ind, col_ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)
    acc = cost_matrix[row_ind, col_ind].sum() / y_true.size
    return acc

def purity_score(y_true, y_pred):
    
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)
    
    n_samples = y_true.size
    purity = 0
    for label in labels_pred:
        idx = np.where(y_pred == label)[0]
        if len(idx) == 0:
            continue
        true_counts = np.bincount(y_true[idx])
        purity += true_counts.max()
    return purity / n_samples

def evaluate_clustering(y_true, y_pred):
    
    acc = clustering_accuracy(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    purity = purity_score(y_true, y_pred)
    
    results = {
        'ACC': acc,
        'NMI': nmi,
        'ARI': ari,
        'F1-score': f1,
        'Precision': precision,
        'Recall': recall,
        'Purity': purity
    }
    return results
