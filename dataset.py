import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import ndimage

# --- Helper Functions for Boundary Loss ---
def compute_distance_map(mask):
    """
    Computes signed distance map (Phi) for Boundary Loss.
    Normalized to [-1, 1] range to prevent Gradient Explosion.
    """
    mask = mask.astype(np.uint8)
    if mask.sum() == 0:
        h, w = mask.shape
        return np.ones((h, w), dtype=np.float32) 
        
    # dist to nearest 0 (for pixels that are 1)
    d_fg = ndimage.distance_transform_edt(mask) 
    # dist to nearest 0 (for pixels that are 0, i.e., distance to 1)
    d_bg = ndimage.distance_transform_edt(1 - mask)
    
    # Combined: Positive outside, Negative inside
    phi = d_bg - d_fg
    
    # CRITICAL: Normalize to [-1, 1] by dividing by a constant (e.g., 20 pixels)
    # This prevents the loss values from becoming huge and causing NaNs.
    phi = np.clip(phi, -20.0, 20.0) / 20.0
    
    return phi.astype(np.float32)

def compute_edge_weight_map(mask, w_edge=5.0, sigma=3.0):
    """
    Creates a weight map that amplifies loss near the boundaries.
    """
    mask = mask.astype(np.uint8)
    if mask.sum() == 0:
        return np.ones_like(mask, dtype=np.float32)
        
    d_fg = ndimage.distance_transform_edt(mask)
    d_bg = ndimage.distance_transform_edt(1 - mask)
    raw_dist = np.abs(d_bg - d_fg)
    
    # Gaussian decay: Weight is highest (1 + w_edge) at dist=0
    weights = 1.0 + w_edge * np.exp(-(raw_dist**2) / (2 * sigma**2))
    return weights.astype(np.float32)

# --- Main Dataset Class ---
class PatchDataset(Dataset):
    def __init__(self, folder):
        self.files = sorted(glob.glob(os.path.join(folder, "patch_*.npz")))
        if len(self.files) == 0:
            raise RuntimeError(f"No patches found in {folder}. Check dataset path!")
        print(f"[Dataset] Loaded {len(self.files)} patches")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        X = data["X"].astype(np.float32) # (8, 256, 256)
        y = data["y"]                    # (256, 256)

        # 1. Clean Data (Handle NaNs)
        X = np.nan_to_num(X, nan=0.0)
        X = np.clip(X, 0.0, 1.0)

        # 2. Calculate Normalized NDVI (Shadow Fix)
        # Assuming Red=Index 3, NIR=Index 7
        red = X[3]
        nir = X[7]
        
        # Avoid div by zero
        ndvi = (nir - red) / (nir + red + 1e-6)
        
        # Normalize from [-1, 1] to [0, 1] for model stability
        ndvi = (ndvi + 1) / 2.0 
        ndvi = np.clip(ndvi, 0.0, 1.0)
        
        # Stack as 9th channel
        ndvi = np.expand_dims(ndvi, axis=0)
        X = np.concatenate([X, ndvi], axis=0)

        # 3. Compute Boundary Maps (CPU)
        dist_map = compute_distance_map(y)
        weight_map = compute_edge_weight_map(y)

        # 4. To Tensor
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        dist_map = torch.tensor(dist_map, dtype=torch.float32)
        weight_map = torch.tensor(weight_map, dtype=torch.float32)

        return X, y, dist_map, weight_map