import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

from model_custom6 import AttentionResUNet
from dataset import PatchDataset

# --- CONFIGURATION ---
CHECKPOINT_PATH = "checkpoints/model_final.pth"
DATA_DIR = Path("data/patches_npz")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def predict_with_tta(model, input_tensor):
    """
    Test-Time Augmentation: Predict on Original + Flip H + Flip V
    """
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(input_tensor)
        p1 = F.softmax(logits, dim=1)[:, 1, :, :]
        
        x_h = torch.flip(input_tensor, [3])
        logits_h, _, _ = model(x_h)
        p2 = F.softmax(logits_h, dim=1)[:, 1, :, :]
        p2 = torch.flip(p2, [2]) 
        
        x_v = torch.flip(input_tensor, [2])
        logits_v, _, _ = model(x_v)
        p3 = F.softmax(logits_v, dim=1)[:, 1, :, :]
        p3 = torch.flip(p3, [1])
        
        p_avg = (p1 + p2 + p3) / 3.0
    return p_avg

def find_optimal_threshold(model, dataset, num_samples=200):
    print("Scanning for Optimal Threshold...")
    model.eval()
    thresholds = np.arange(0.1, 0.95, 0.05)
    best_dice = 0.0
    best_thresh = 0.5
    
    indices = random.sample(range(len(dataset)), min(len(dataset), num_samples))
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for idx in indices:
            img, mask, _, _ = dataset[idx] # Unpack 4 items
            input_tensor = img.unsqueeze(0).to(DEVICE)
            
            # Use TTA even for threshold finding
            pred_prob = predict_with_tta(model, input_tensor)
            
            all_probs.append(pred_prob.cpu().numpy().flatten())
            all_targets.append(mask.numpy().flatten())
            
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    
    for t in thresholds:
        preds = (all_probs > t).astype(np.uint8)
        tp = np.sum((preds == 1) & (all_targets == 1))
        fp = np.sum((preds == 1) & (all_targets == 0))
        fn = np.sum((preds == 0) & (all_targets == 1))
        
        p = tp / (tp + fp + 1e-6)
        r = tp / (tp + fn + 1e-6)
        dice = 2 * (p * r) / (p + r + 1e-6)
        
        if dice > best_dice:
            best_dice = dice
            best_thresh = t
            
    print(f"Optimal Threshold Found: {best_thresh:.2f} (Dice: {best_dice:.4f})")
    return best_thresh

def evaluate_metrics():
    print(f"Loading Dataset from {DATA_DIR}...")
    dataset = PatchDataset(DATA_DIR)
    
    print(f"Loading Model from {CHECKPOINT_PATH}...")
    model = AttentionResUNet(in_channels=9, out_channels=2).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval()
    
    # Dynamic Thresholding
    OPTIMAL_THRESHOLD = find_optimal_threshold(model, dataset)
    
    print(f"Evaluating Full Dataset with TTA @ Threshold {OPTIMAL_THRESHOLD}...")
    
    TP_total = FP_total = FN_total = 0
    
    # Evaluate on a subset or full set (here full set for final numbers)
    # Using TTA is slow, so be patient.
    for idx in tqdm(range(len(dataset))):
        image, mask, _, _ = dataset[idx]
        input_tensor = image.unsqueeze(0).to(DEVICE)
        
        pred_prob = predict_with_tta(model, input_tensor)
        pred_mask = (pred_prob > OPTIMAL_THRESHOLD).cpu().numpy().astype(np.uint8)
        gt_mask = mask.numpy().astype(np.uint8)
        
        p_flat = pred_mask.flatten()
        g_flat = gt_mask.flatten()
        
        TP_total += np.sum((p_flat == 1) & (g_flat == 1))
        FP_total += np.sum((p_flat == 1) & (g_flat == 0))
        FN_total += np.sum((p_flat == 0) & (g_flat == 1))
        
    epsilon = 1e-6
    precision = TP_total / (TP_total + FP_total + epsilon)
    recall = TP_total / (TP_total + FN_total + epsilon)
    dice_score = 2 * (precision * recall) / (precision + recall + epsilon)
    iou_score = TP_total / (TP_total + FP_total + FN_total + epsilon)
    
    print("\n" + "="*40)
    print("       FINAL PERFORMANCE REPORT       ")
    print("="*40)
    print(f"True Positives (Pixels): {TP_total}")
    print(f"False Positives (Pixels): {FP_total}")
    print(f"False Negatives (Pixels): {FN_total}")
    print("-" * 40)
    print(f"Precision: {precision:.4f}  ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f}  ({recall*100:.2f}%)")
    print("-" * 40)
    print(f"Dice Score (F1): {dice_score:.4f}")
    print(f"IoU Score:       {iou_score:.4f}")
    print("="*40)

if __name__ == "__main__":
    evaluate_metrics()