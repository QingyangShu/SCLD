import numpy as np
from readdata import load_binary_multiview_data
from initial import initialize_all
from admm import admm_update
from propagate_labels import propagate_labels
from sklearn.metrics import f1_score

# ----------------------------
# Configuration parameters
# ----------------------------
mat_path = "Wikipedia.mat"
class_pos = [1]
class_neg = [10]
known_pos_ratio = 0.2
seed = 42
n_clusters = 2
max_iter = 100
eta = 0.99
lambda1 = 10   
lambda2 = 1   
lambda3 = 0.001   
lambda4 = 1000   
r = 2           

# ----------------------------
# Data loading
# ----------------------------
X_views, y_binary, known_pos_idx, unknown_idx = load_binary_multiview_data(
    mat_path, class_pos, class_neg,
    known_pos_ratio=known_pos_ratio,
    seed=seed
)

print("=== Data reading completed ===")
print(f"Number of views: {len(X_views)}")
print(f"Number of samples: {len(y_binary)}")
print(f"Number of positive samples: {np.sum(y_binary == 1)}")
print(f"Number of negative samples: {np.sum(y_binary == 0)}")
for v, Xv in enumerate(X_views):
    print(f"View {v} feature dimensions: {Xv.shape}")

# ----------------------------
# Initialize
# ----------------------------
alpha, S_list, Z, F, L_z = initialize_all(X_views, n_clusters, known_pos_idx=known_pos_idx)

# ----------------------------
# ADMM
# ----------------------------
alpha, Z, F, loss_list = admm_update(
    alpha, Z, F, S_list, known_pos_idx,
    n_clusters=n_clusters,
    lambda1=lambda1, lambda2=lambda2, lambda3=lambda3, lambda4=lambda4,
    r=r, max_iter=max_iter, tol=1e-6
)

# ----------------------------
# Label confidence propagation
# ----------------------------
H_final = propagate_labels(Z, n_classes=n_clusters, known_pos_idx=known_pos_idx, eta=eta, max_iter=200)
n_samples = H_final.shape[0]
n_known_pos = len(known_pos_idx)
known_pos_ratio_in_true = n_known_pos / np.sum(y_binary == 1)
known_pos_ratio_in_all = n_known_pos / n_samples
estimated_pos_ratio = known_pos_ratio_in_all / known_pos_ratio_in_true
n_pos_estimated = int(round(estimated_pos_ratio * n_samples))
H_score = H_final[:, 0]
sorted_idx = np.argsort(H_score)[::-1]
pos_idx = sorted_idx[:n_pos_estimated]
y_pred_final = np.zeros(n_samples, dtype=int)
y_pred_final[pos_idx] = 1
known_pos_correct = np.sum(y_pred_final[known_pos_idx] == 1)

# ----------------------------
# Evaluate
# ----------------------------
f1 = f1_score(y_binary, y_pred_final)
print(f"F1-score: {f1:.4f}")


