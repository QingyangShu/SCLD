# initial.py
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import eigh

def initialize_alpha(n_views):
    
    alpha = np.ones(n_views) / n_views
    return alpha

def initialize_S(X_views, k=5, sigma=None, normalize_features=True):
    
    n_views = len(X_views)
    n_samples = X_views[0].shape[0]
    S_list = []

    for v in range(n_views):
        Xv = np.asarray(X_views[v], dtype=np.float32)
        
        
        if normalize_features:
            from sklearn.preprocessing import normalize as sk_normalize
            Xv = sk_normalize(Xv, norm='l2', axis=1)

        
        k_adaptive = min(k, max(5, int(np.sqrt(n_samples))))
        n_neighbors = min(k_adaptive + 1, n_samples)
        
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', n_jobs=-1).fit(Xv)
        distances, indices = nbrs.kneighbors(Xv)

        
        if sigma is None:
            k_th_distances = distances[:, min(k, n_neighbors-1)]
            sigma_v = float(np.median(k_th_distances))
            if sigma_v < 1e-6:
                sigma_v = 1.0
        else:
            sigma_v = sigma

        
        S = np.zeros((n_samples, n_samples), dtype=np.float32)

        
        for i in range(n_samples):
            idxs = indices[i]
            dists = distances[i]
            mask = idxs != i
            if np.any(mask):
                valid_idx = idxs[mask]
                valid_dists = dists[mask]
                
                S[i, valid_idx] = np.exp(- (valid_dists**2) / (2 * sigma_v**2))

        
        S = np.maximum(S, S.T)
        
        
        np.fill_diagonal(S, 0.0)
        
        
        row_sums = S.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        S = S / row_sums
        
        S_list.append(S)

    return S_list

def initialize_Z(S_list, known_pos_idx=None, mu=0.2):
   
    n_samples = S_list[0].shape[0]
    V = len(S_list)
    
   
    Z = np.zeros((n_samples, n_samples), dtype=np.float32)
    for v in range(V):
        Z += S_list[v]
    Z = Z / V
    
    np.fill_diagonal(Z, 0.0)

   
    if known_pos_idx is not None and len(known_pos_idx) > 1:
        kp = np.asarray(known_pos_idx, dtype=np.int32)
        m = kp.size
        if m > 1:
            
            sub = Z[np.ix_(kp, kp)]
            
           
            offdiag = np.ones((m, m), dtype=np.float32) - np.eye(m, dtype=np.float32)
            sub = sub + mu * offdiag
            
           
            sub = np.clip(sub, 0, 1)
            np.fill_diagonal(sub, 0.0)
            
            Z[np.ix_(kp, kp)] = sub


    Z = (Z + Z.T) / 2.0
    
    
    Z_max = Z.max()
    if Z_max > 1e-10:
        Z = Z / Z_max
    
    Z = Z.astype(np.float32)
    return Z

def compute_Laplacian(Z):
    
    row_sum = Z.sum(axis=1).astype(np.float32)
    D = np.diag(row_sum)
    L_z = D - Z
    return L_z

def initialize_F(L_z, n_clusters):
    
    n_samples = L_z.shape[0]
    
    
    L_z_reg = L_z + 1e-8 * np.eye(n_samples, dtype=np.float32)
    
    try:
       
        eigvals, eigvecs = eigh(L_z_reg)
        
        
        F = eigvecs[:, :n_clusters].astype(np.float32)
        
        
        row_norms = np.linalg.norm(F, axis=1, keepdims=True)
        row_norms = np.maximum(row_norms, 1e-10)
        F = F / row_norms
        
        
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = kmeans.fit_predict(F)
        
        
        F_onehot = np.zeros((n_samples, n_clusters), dtype=np.float32)
        F_onehot[np.arange(n_samples), labels] = 1.0
        
        
        from sklearn.preprocessing import normalize as sk_normalize
        F = sk_normalize(F_onehot, norm='l2', axis=0).astype(np.float32)
        
    except Exception as e:
        print(f"Warning: Eigen decomposition failed during F initialization: {e}")
        F = np.random.randn(n_samples, n_clusters).astype(np.float32)
        from sklearn.preprocessing import normalize as sk_normalize
        F = sk_normalize(F, norm='l2', axis=0).astype(np.float32)
    
    return F

def initialize_all(X_views, n_clusters, known_pos_idx=None, k=10, mu=0.2):
    
    n_views = len(X_views)
    n_samples = X_views[0].shape[0]

    
    k_adaptive = min(k, max(5, int(np.sqrt(n_samples))))

    alpha = initialize_alpha(n_views)
    S_list = initialize_S(X_views, k=k_adaptive, normalize_features=True)
    Z = initialize_Z(S_list, known_pos_idx=known_pos_idx, mu=mu)
    L_z = compute_Laplacian(Z)
    F = initialize_F(L_z, n_clusters)

    return alpha, S_list, Z, F, L_z
