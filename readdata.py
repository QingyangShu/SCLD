import scipy.io
import numpy as np

def load_binary_multiview_data(mat_path, class_pos, class_neg, known_pos_ratio=0.2, seed=42):
    
    mat = scipy.io.loadmat(mat_path)
    X_cell = mat['X']
    y = mat['y'].squeeze()

    
    if not isinstance(class_pos, (list, tuple, np.ndarray)):
        class_pos = [class_pos]
    if not isinstance(class_neg, (list, tuple, np.ndarray)):
        class_neg = [class_neg]

    
    is_pos = np.isin(y, class_pos)
    is_neg = np.isin(y, class_neg)
    selected_mask = is_pos | is_neg
    selected_idx = np.where(selected_mask)[0]
    
    y_selected = y[selected_idx]
    y_binary = np.zeros(len(y_selected), dtype=int)


    y_binary[np.isin(y_selected, class_pos)] = 1  

    
    X_views = [np.asarray(X_cell[i, 0][selected_idx, :], dtype=np.float32) for i in range(X_cell.shape[0])]
    n_views = len(X_views)
    n_samples = len(selected_idx)
    rng = np.random.default_rng(seed)
    
    
    pos_idx = np.where(y_binary == 1)[0]
    n_known = max(1, int(len(pos_idx) * known_pos_ratio))
    known_pos_idx = rng.choice(pos_idx, n_known, replace=False)

    
    unknown_idx = np.setdiff1d(np.arange(n_samples), known_pos_idx)

    
    y_binary = np.asarray(y_binary, dtype=np.int8)
    known_pos_idx = np.asarray(known_pos_idx, dtype=np.int32)
    unknown_idx = np.asarray(unknown_idx, dtype=np.int32)

    return X_views, y_binary, known_pos_idx, unknown_idx

