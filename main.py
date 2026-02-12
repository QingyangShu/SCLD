import numpy as np
from readdata import load_binary_multiview_data
from initial import initialize_all
from admm import admm_update
from evaluate import evaluate_clustering
from propagate_labels import propagate_labels, select_adaptive_threshold
from sklearn.metrics import f1_score


mat_path = "C:/Users/SHU Qing'yang/Desktop/Multi-view datasets/Wikipedia.mat"
class_pos = [1]
class_neg = [10]
known_pos_ratio = 0.2
seed = 42
n_clusters = 2
max_iter = 100  
eta = 0.99  
lambda1 = 1   
lambda2 = 1   
lambda3 = 0.001   
lambda4 = 100   
r = 2           


X_views, y_binary, known_pos_idx, unknown_idx = load_binary_multiview_data(
    mat_path, class_pos, class_neg,
    known_pos_ratio=known_pos_ratio,
    seed=seed
)
alpha, S_list, Z, F, L_z = initialize_all(X_views, n_clusters, known_pos_idx=known_pos_idx)
alpha, Z, F, loss_list = admm_update(
    alpha, Z, F, S_list, known_pos_idx,
    n_clusters=n_clusters,
    lambda1=lambda1, lambda2=lambda2, lambda3=lambda3, lambda4=lambda4,
    r=r, max_iter=max_iter, tol=1e-6
)
y_pred_cluster = np.argmax(F, axis=1)
results_cluster = evaluate_clustering(y_binary, y_pred_cluster)
H_final = propagate_labels(Z, n_classes=n_clusters, known_pos_idx=known_pos_idx, eta=eta, max_iter=200)
n_samples = H_final.shape[0]
H_score = H_final[:, 0]
tau_optimal, threshold_scores, tau_candidates = select_adaptive_threshold(
    Z, H_score, n_candidates=100
)
y_pred_final = np.zeros(n_samples, dtype=int)
y_pred_final[H_score >= tau_optimal] = 1
n_known_pos = len(known_pos_idx)
known_pos_correct = np.sum(y_pred_final[known_pos_idx] == 1)
f1 = f1_score(y_binary, y_pred_final)
print(f"F1-score: {f1:.4f}")

