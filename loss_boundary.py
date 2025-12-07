import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage

EPS = 1e-7

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.45, beta=0.55, gamma=4/3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits, target, weight_map=None):
        probs = torch.sigmoid(logits).view(-1)
        target = target.view(-1).float()
        
        if weight_map is not None:
            w = weight_map.view(-1).float()
            # Apply spatial weighting to the probabilities
            # This emphasizes edge pixels in the TP/FP/FN calculation
            probs = probs * w
            target = target * w # Scale target effectively by importance

        TP = (probs * target).sum()
        FP = ((1 - target) * probs).sum()
        FN = (target * (1 - probs)).sum()

        TI = (TP + EPS) / (TP + self.alpha * FP + self.beta * FN + EPS)
        return (1 - TI) ** self.gamma

class SoftFPRatioPen(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, target):
        probs = torch.sigmoid(logits).view(-1)
        target = target.view(-1).float()
        neg_mask = (1.0 - target)
        numerator = (neg_mask * probs).sum()
        denom = neg_mask.sum() + EPS
        return numerator / denom

class BoundaryLoss(nn.Module):
    """
    Penalizes predictions based on distance from the ground truth.
    Far away false positives get a HUGE penalty.
    Nearby false positives get a small penalty.
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits, signed_distance_map):
        probs = torch.sigmoid(logits)
        # signed_distance_map is Positive outside GT, Negative inside GT.
        # We want P to be low where Distance is Positive (outside).
        # We want P to be high where Distance is Negative (inside).
        loss_map = probs * signed_distance_map
        return loss_map.mean()

class CombinedBoundaryLoss(nn.Module):
    def __init__(self, alpha=0.45, beta=0.55, gamma=4/3, lambda_fp=0.2, mu_boundary=0.1):
        super().__init__()
        self.ft = FocalTverskyLoss(alpha, beta, gamma)
        self.softfp = SoftFPRatioPen()
        self.boundary = BoundaryLoss()
        self.lambda_fp = lambda_fp
        self.mu_boundary = mu_boundary

    def forward(self, logits, target, dist_map, weight_map):
        # Flatten logic handled inside sub-components
        logits_pos = logits[:, 1, :, :]
        
        l_ft = self.ft(logits_pos, target, weight_map)
        l_fp = self.softfp(logits_pos, target)
        l_bound = self.boundary(logits_pos, dist_map)
        
        return l_ft + (self.lambda_fp * l_fp) + (self.mu_boundary * l_bound)

# --- Helper Functions for Dataset ---
def compute_distance_map(mask):
    """
    Computes signed distance map.
    Positive distance outside the mask.
    Negative distance inside the mask.
    """
    mask = mask.astype(np.uint8)
    if mask.sum() == 0:
        # If empty mask, distance to FG is infinite (or large)
        # Distance to BG is 0 everywhere.
        # Signed dist = 0 - large = -large (Actually we want positive everywhere outside)
        # Correct logic: if no FG, everywhere is BG. Dist to FG is huge.
        # We want penalty everywhere P>0. So phi should be positive everywhere.
        h, w = mask.shape
        return np.ones((h, w), dtype=np.float32) * 100.0 # Large constant penalty
        
    d_fg = ndimage.distance_transform_edt(mask) 
    d_bg = ndimage.distance_transform_edt(1 - mask)
    
    # phi > 0 outside (d_bg > 0)
    # phi < 0 inside (d_fg > 0)
    phi = d_bg - d_fg
    return phi.astype(np.float32)

def compute_edge_weight_map(mask, w_edge=5.0, sigma=3.0):
    """
    Creates a weight map that is high (1 + w_edge) near boundaries
    and low (1.0) elsewhere.
    """
    # Distance to nearest boundary
    phi = compute_distance_map(mask)
    dist_to_boundary = np.abs(phi)
    
    # Gaussian decay
    weights = 1.0 + w_edge * np.exp(-(dist_to_boundary**2) / (2 * sigma**2))
    return weights.astype(np.float32)