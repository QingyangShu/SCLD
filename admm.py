import numpy as np
from scipy.linalg import eigh
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

def simplex_projection(y):
    n = len(y)
    if n == 0:
        return y
    
    u = np.sort(y)[::-1]
    
    cumsum_u = np.cumsum(u)
    
    rho = 0
    for j in range(n):
        if u[j] + (1.0 - cumsum_u[j]) / (j + 1) > 0:
            rho = j
    
    theta = (cumsum_u[rho] - 1.0) / (rho + 1)
    
    z = np.maximum(y - theta, 0.0)
    
    return z

def batch_simplex_projection(Y):

    n, m = Y.shape
    Z = np.zeros_like(Y)
    
    for j in range(m):
        col = Y[:, j].copy()
        col[j] = 0.0  
        Z[:, j] = simplex_projection(col)
    
    return Z

def compute_loss(alpha, Z, F, S_list, L_z, lambda1, lambda2, lambda3, lambda4, known_pos_idx, r=2):

    V = len(S_list)
    loss = 0.0
    weights = alpha ** r  
    
    
    for v in range(V):
        diff = Z - S_list[v]
        
        loss += weights[v] * (np.sum(diff * diff) + lambda1 * np.sum(S_list[v] * (L_z @ S_list[v])))

    
    loss += lambda2 * np.sum(F * (L_z @ F))
    loss += lambda3 * np.sum(Z * Z)

    
    if known_pos_idx is not None and len(known_pos_idx) > 1:
        kp = np.asarray(known_pos_idx, dtype=np.int32)
        sub = Z[np.ix_(kp, kp)]
        m = kp.size
        if m > 1:
            loss += lambda4 * (np.sum((1 - sub) ** 2) - m)

    return loss

def admm_update(alpha, Z, F, S_list, known_pos_idx, n_clusters=2, lambda1=1.0, lambda2=1.0, lambda3=1.0, lambda4=1.0, r=2, max_iter=50, tol=1e-5, rho=1.0):
    V = len(S_list)
    n_samples = S_list[0].shape[0]

    loss_list = []
    
    G_list = []
    for v in range(V):
        S_v = S_list[v]
        row_norms_sq = np.sum(S_v ** 2, axis=1, keepdims=True)
        G_v = row_norms_sq + row_norms_sq.T - 2.0 * (S_v @ S_v.T)
        G_list.append(G_v.astype(np.float32))
    
    M = np.zeros((n_samples, n_samples), dtype=np.float32)
    if known_pos_idx is not None and len(known_pos_idx) > 1:
        kp = np.asarray(known_pos_idx, dtype=np.int32)
        
        idx_grid = np.meshgrid(kp, kp, indexing='ij')
        mask = idx_grid[0] != idx_grid[1]
        M[idx_grid[0][mask], idx_grid[1][mask]] = 1.0
    
    
    const_denom = 2.0 * lambda3
    const_numerator_M = 2.0 * lambda4 * M
    const_denom_M = 2.0 * lambda4 * M
    
    L_z = np.diag(Z.sum(axis=1)) - Z
    initial_loss = compute_loss(alpha, Z, F, S_list, L_z, lambda1, lambda2, lambda3, lambda4, known_pos_idx, r)
    loss_list.append(float(initial_loss))
    
    for it in range(max_iter):
        
        weights = alpha ** r  
        
        
        S_bar = np.tensordot(weights, S_list, axes=([0], [0])).astype(np.float32)
        
        
        G_bar = lambda1 * np.tensordot(weights, G_list, axes=([0], [0])).astype(np.float32)
        
        
        row_norms_sq_F = np.sum(F ** 2, axis=1, keepdims=True)
        G_F = row_norms_sq_F + row_norms_sq_F.T - 2.0 * (F @ F.T)
        G_bar += lambda2 * G_F.astype(np.float32)
        
        
        sum_weights = np.sum(weights)
        numerator = 2.0 * S_bar - 0.5 * G_bar + const_numerator_M
        denominator = 2.0 * sum_weights + const_denom + const_denom_M
        
      
        Z_star = numerator / (denominator + 1e-10)
        np.fill_diagonal(Z_star, 0.0)
        
        
        Z = batch_simplex_projection(Z_star)
        Z = Z.astype(np.float32)

        
        D = np.diag(Z.sum(axis=1))
        L_z = D - Z

        
        try:
            
            L_z_reg = L_z + 1e-8 * np.eye(n_samples, dtype=np.float32)
            
            if n_samples > 200 and n_clusters < n_samples - 1:
                
                Lz_sparse = sp.csr_matrix(L_z_reg)
                eigvals, eigvecs = eigsh(Lz_sparse, k=n_clusters, which='SM', maxiter=1000)
                F = eigvecs.astype(np.float32)
            else:
                eigvals, eigvecs = eigh(L_z_reg)
                F = eigvecs[:, :n_clusters].astype(np.float32)
            
            
            for i in range(n_clusters):
                norm = np.linalg.norm(F[:, i])
                if norm > 1e-10:
                    F[:, i] = F[:, i] / norm
            
        except Exception as e:
            print(f"error in eigen decomposition: {e}")
            

        
        tmp = np.zeros(V, dtype=np.float32)
        for v in range(V):
            diff = Z - S_list[v]
            tmp[v] = np.sum(diff * diff) + lambda1 * np.sum(S_list[v] * (L_z @ S_list[v]))

        
        tmp = np.clip(tmp, 1e-10, 1e10)
        
        if r == 1:
            alpha = np.ones(V, dtype=np.float32) / V
        else:
            alpha = (1.0 / tmp) ** (1.0 / (r - 1.0))
            alpha = np.nan_to_num(alpha, nan=1.0/V, posinf=1.0, neginf=0.0)
            alpha_sum = np.sum(alpha)
            if alpha_sum > 1e-10:
                alpha = alpha / alpha_sum
            else:
                alpha = np.ones(V, dtype=np.float32) / V

        loss = compute_loss(alpha, Z, F, S_list, L_z, lambda1, lambda2, lambda3, lambda4, known_pos_idx, r)
        loss_list.append(float(loss))
        
        if (it + 1) % 10 == 0 or it == 0:
            loss_change = loss_list[-1] - loss_list[-2] if len(loss_list) > 1 else 0
        
        if it > 0:
            abs_change = abs(loss_list[-1] - loss_list[-2])
            rel_change = abs_change / (abs(loss_list[-2]) + 1e-10)
            
            if rel_change < tol:
                break
            
            if len(loss_list) > 3:
                if loss_list[-1] > loss_list[-2] and loss_list[-2] > loss_list[-3]:
                    break

    return alpha, Z, F, loss_list

