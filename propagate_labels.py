import numpy as np

def compute_structural_consistency_score(Z, H_pos_confidence, tau, epsilon=1e-10):
    
    pos_mask = H_pos_confidence >= tau
    n_pos = pos_mask.sum()
    n_total = len(pos_mask)
    
    if n_pos == 0 or n_pos == n_total or n_pos < 5 or n_pos > n_total * 0.9:
        return -np.inf
    
    Z_pos = Z[pos_mask][:, pos_mask]
    internal_connectivity = Z_pos.sum()
    
    Z_cross = Z[pos_mask][:, ~pos_mask]
    external_connectivity = Z_cross.sum()
    
    score = internal_connectivity / (external_connectivity + epsilon)
    
    score = score / n_pos
    
    return score


def select_adaptive_threshold(Z, H_pos_confidence, n_candidates=100, min_tau=None, max_tau=None):
    if min_tau is None:
        min_tau = np.percentile(H_pos_confidence, 50)  
    if max_tau is None:
        max_tau = np.percentile(H_pos_confidence, 99) 
    
    tau_candidates = np.linspace(min_tau, max_tau, n_candidates)
    
    scores = []
    best_score = -np.inf
    tau_optimal = min_tau
    
    for tau in tau_candidates:
        score = compute_structural_consistency_score(Z, H_pos_confidence, tau)
        scores.append(score)
        
        if score > best_score:
            best_score = score
            tau_optimal = tau
    
    n_pos_at_optimal = (H_pos_confidence >= tau_optimal).sum()
    
    return tau_optimal, scores, tau_candidates


def propagate_labels(Z, n_classes, known_pos_idx, eta=0.8, max_iter=100, tol=1e-6):

    n_samples = Z.shape[0]

    D = np.diag(Z.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / (np.sqrt(np.diag(D)) + 1e-10))
    Z_norm = D_inv_sqrt @ Z @ D_inv_sqrt
    Z_norm = Z_norm.astype(np.float32)

    mask = np.ones(n_samples, dtype=np.float32)
    mask[known_pos_idx] = 0.0
    M = np.diag(mask)  

    H0 = np.full((n_samples, n_classes), 1.0 / n_classes, dtype=np.float32)
    
    H0[known_pos_idx, :] = 0.0
    H0[known_pos_idx, 0] = 1.0  

    H_prev = H0.copy()
    H = H0.copy()

    I_minus_M_eta = np.eye(n_samples, dtype=np.float32) - eta * M


    best_H = H.copy()
    best_diff = float('inf')
    
    for it in range(max_iter):
    
        propagated = M @ (eta * (Z_norm.T @ H_prev))  
        fixed = I_minus_M_eta @ H0  
        H = propagated + fixed

        row_sums = H.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        H = H / row_sums
        
        H = np.clip(H, 0, 1)
        
        H[known_pos_idx, :] = 0.0
        H[known_pos_idx, 0] = 1.0

        diff = np.max(np.abs(H - H_prev))
        
        if diff < best_diff:
            best_diff = diff
            best_H = H.copy()
        
        if diff < tol:
            if it < 10:  
                pass
            else:
                break
        
        H_prev = H.copy()


    return best_H
