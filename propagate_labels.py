import numpy as np

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
