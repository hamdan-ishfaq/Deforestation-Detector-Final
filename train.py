import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import time
import random
import numpy as np

# Import from your existing files
from dataset import PatchDataset
from model_custom6 import AttentionResUNet   

device = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-6

# ==========================================
#        ROBUST LOSS FUNCTIONS
# ==========================================

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.45, beta=0.55, gamma=4/3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits, target, weight_map=None):
        probs = torch.sigmoid(logits)
        probs = probs.reshape(-1)
        target = target.reshape(-1).float()
        
        if weight_map is not None:
            w = weight_map.reshape(-1).float()
            probs = probs * w
            target = target * w 

        TP = (probs * target).sum()
        FP = ((1 - target) * probs).sum()
        FN = (target * (1 - probs)).sum()

        numerator = TP + EPS
        denominator = TP + (self.alpha * FP) + (self.beta * FN) + EPS
        TI = numerator / denominator
        TI = torch.clamp(TI, 0.0, 1.0) # Prevent NaN
        
        return (1 - TI) ** self.gamma

class SoftFPRatioPen(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, target):
        probs = torch.sigmoid(logits).reshape(-1)
        target = target.reshape(-1).float()
        
        neg_mask = (1.0 - target)
        numerator = (neg_mask * probs).sum()
        denom = neg_mask.sum() + EPS
        
        return numerator / denom

class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, signed_distance_map):
        probs = torch.sigmoid(logits)
        # Dist map is normalized [-1, 1], so no explosion
        loss_map = probs * signed_distance_map
        return loss_map.mean()

class CombinedBoundaryLoss(nn.Module):
    def __init__(self, alpha=0.45, beta=0.55, gamma=4/3, lambda_fp=0.2):
        super().__init__()
        self.ft = FocalTverskyLoss(alpha, beta, gamma)
        self.softfp = SoftFPRatioPen()
        self.boundary = BoundaryLoss()
        self.lambda_fp = lambda_fp

    def forward(self, logits, target, dist_map, weight_map, current_mu_boundary):
        logits_pos = logits[:, 1, :, :]
        
        l_ft = self.ft(logits_pos, target, weight_map)
        l_fp = self.softfp(logits_pos, target)
        
        total = l_ft + (self.lambda_fp * l_fp)
        
        # Dynamic Boundary Weight (Curriculum Learning)
        if current_mu_boundary > 0:
            l_bound = self.boundary(logits_pos, dist_map)
            total = total + (current_mu_boundary * l_bound)
            
        return total

# ==========================================
#        EARLY STOPPING (30 EPOCH GRACE)
# ==========================================
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, grace_period=30, path='checkpoints/model_final.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.grace_period = grace_period
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model, epoch):
        # Ignore checks during grace period, but SAVE best model if found
        if epoch <= self.grace_period:
            if self.best_loss is None or val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(val_loss, model)
            return

        # Start actual counting
        if val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.best_loss = val_loss

# ==========================================
#              TRAINING LOOP
# ==========================================

patch_dir = Path("data/patches_npz") 
dataset = PatchDataset(patch_dir)

# Full dataset for training to maximize performance
train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0) 

print(f"Initializing Attention model (9 Channels) on {device}...")
model = AttentionResUNet(in_channels=9, out_channels=2).to(device) 

seg_loss_fn = CombinedBoundaryLoss(alpha=0.45, beta=0.55, lambda_fp=0.2)
percent_loss_fn = nn.MSELoss()

opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=2, factor=0.5)

# --- CONFIGURATION ---
MAX_EPOCHS = 100
early_stopper = EarlyStopping(patience=10, grace_period=30, path="checkpoints/model_final.pth")

def apply_radiometric_augmentation(x):
    if random.random() > 0.5: 
        factor = random.uniform(0.7, 1.3)
        x = x * factor
    if random.random() > 0.5: 
        factor = random.uniform(0.8, 1.2)
        mean = x.mean(dim=(2, 3), keepdim=True)
        x = (x - mean) * factor + mean
    return torch.clamp(x, 0.0, 1.0)

def run_epoch(loader, epoch_idx):
    model.train() 
    total_loss = 0.0
    
    # Boundary Loss Warmup Schedule
    if epoch_idx <= 3: mu_boundary = 0.0
    elif epoch_idx <= 10: mu_boundary = 0.05
    else: mu_boundary = 0.1

    print(f"Epoch {epoch_idx} | Boundary Weight: {mu_boundary}")

    TP_sum = FP_sum = TN_sum = FN_sum = 0

    for batch_idx, (x, y, dist_map, weight_map) in enumerate(loader):
        x, y = x.to(device), y.to(device).long()
        dist_map = dist_map.to(device)
        weight_map = weight_map.to(device)
        
        # Safety Check
        if torch.isnan(x).any() or torch.isnan(dist_map).any():
            continue

        if y.dim() == 4: y = y.squeeze(1)

        # Augmentations
        if random.random() > 0.5: 
            x = torch.flip(x, [3]); y = torch.flip(y, [2]); dist_map = torch.flip(dist_map, [2]); weight_map = torch.flip(weight_map, [2])
        if random.random() > 0.5: 
            x = torch.flip(x, [2]); y = torch.flip(y, [1]); dist_map = torch.flip(dist_map, [1]); weight_map = torch.flip(weight_map, [1])
        
        x = apply_radiometric_augmentation(x)

        opt.zero_grad()

        seg_logits, aux_seg, percent_preds = model(x)

        loss_seg = seg_loss_fn(seg_logits, y, dist_map, weight_map, mu_boundary)

        if aux_seg is not None:
            aux_seg_resized = F.interpolate(aux_seg, size=y.shape[1:], mode='bilinear', align_corners=True)
            aux_loss = seg_loss_fn.ft(aux_seg_resized[:,1,:,:].unsqueeze(1), y)
            loss_seg = loss_seg + 0.3 * aux_loss

        B = y.size(0)
        def_frac = y.reshape(B, -1).float().mean(dim=1)
        loss_percent = percent_loss_fn(percent_preds, def_frac.unsqueeze(1).to(device))
        
        loss = loss_seg + (0.2 * loss_percent)

        if torch.isnan(loss):
            print(f"NaN Loss in Batch {batch_idx}. Skipping.")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        total_loss += loss.item()

        with torch.no_grad():
            _, preds = torch.max(seg_logits, 1)
            y_flat = y.reshape(-1); p_flat = preds.reshape(-1)
            TP_sum += ((p_flat == 1) & (y_flat == 1)).sum().item()
            FP_sum += ((p_flat == 1) & (y_flat == 0)).sum().item()
            FN_sum += ((p_flat == 0) & (y_flat == 1)).sum().item()
            TN_sum += ((p_flat == 0) & (y_flat == 0)).sum().item()

    avg_loss = total_loss / len(loader)
    eps = 1e-6
    rec = TP_sum / (TP_sum + FN_sum + eps)
    prec = TP_sum / (TP_sum + FP_sum + eps)
    
    return avg_loss, rec, prec, TP_sum, FP_sum, TN_sum, FN_sum

print(f"Starting Long-Run Training (Max {MAX_EPOCHS} Epochs)...\n")
Path("checkpoints").mkdir(exist_ok=True)

for epoch in range(1, MAX_EPOCHS + 1):
    start = time.time()
    
    loss, rec, prec, tp, fp, tn, fn = run_epoch(train_loader, epoch)
    
    scheduler.step(rec)
    
    # Early Stopping Check (Will track best model)
    early_stopper(loss, model, epoch)
    
    print(f"TRAIN: Loss={loss:.4f} | Rec={rec:.4f} | Prec={prec:.4f}")
    print(f"MATRIX: TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"Time: {time.time() - start:.1f}s")
    print("-" * 40)
    
    if early_stopper.early_stop:
        print("Early stopping triggered! Training complete.")
        break

print("Done. Final Model saved as checkpoints/model_final.pth")