import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import random

from model_custom6 import AttentionResUNet
from dataset import PatchDataset

CHECKPOINT_PATH = "checkpoints/model_v6_boundary.pth"
DATA_DIR = Path("data/patches_npz")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def clean_mask(mask_prob, threshold, min_object_size=50):
    mask_bin = (mask_prob > threshold).astype(np.uint8)
    kernel = np.ones((3,3), np.uint8)
    mask_clean = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel, iterations=1)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_object_size:
            mask_clean[labels == i] = 0
            
    return mask_clean

def find_optimal_threshold(model, dataset, num_samples=100):
    print("Scanning for optimal threshold...")
    model.eval()
    thresholds = np.arange(0.1, 0.95, 0.05)
    best_f1 = 0.0
    best_thresh = 0.5
    
    # Simple sampling
    indices = random.sample(range(len(dataset)), min(len(dataset), num_samples))
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for idx in indices:
            # Unpack 4 items (we only need image and mask)
            img, mask, _, _ = dataset[idx]
            
            input_tensor = img.unsqueeze(0).to(DEVICE)
            logits, _, _ = model(input_tensor)
            probs = F.softmax(logits, dim=1)
            pred_prob = probs[0, 1].cpu().numpy()
            
            all_probs.append(pred_prob.flatten())
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
        f1 = 2 * (p * r) / (p + r + 1e-6)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            
    print(f"Optimal Threshold: {best_thresh:.2f} (F1: {best_f1:.4f})")
    return best_thresh

def visualize_results(num_samples=5):
    dataset = PatchDataset(DATA_DIR)
    
    print(f"Loading model from {CHECKPOINT_PATH}...")
    model = AttentionResUNet(in_channels=9, out_channels=2).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval()
    
    best_thresh = find_optimal_threshold(model, dataset)
    
    print(f"Visualizing {num_samples} random samples...")
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    viz_indices = []
    for idx in indices:
        _, mask, _, _ = dataset[idx] # Unpack 4
        if mask.sum() > 50: 
            viz_indices.append(idx)
        if len(viz_indices) >= num_samples:
            break
            
    for idx in viz_indices:
        image, mask, _, _ = dataset[idx] # Unpack 4
        input_tensor = image.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits, _, _ = model(input_tensor)
            probs = F.softmax(logits, dim=1)
            pred_prob = probs[0, 1].cpu().numpy()
            
        gt_mask = mask.squeeze().numpy()
        input_img_viz = image[3].numpy() 
        
        cleaned_pred = clean_mask(pred_prob, threshold=best_thresh)
        
        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        ax[0].imshow(input_img_viz, cmap='gray'); ax[0].set_title("Input Red Band"); ax[0].axis('off')
        ax[1].imshow(gt_mask, cmap='gray'); ax[1].set_title("Ground Truth"); ax[1].axis('off')
        ax[2].imshow(pred_prob, cmap='jet', vmin=0, vmax=1); ax[2].set_title("Raw AI Prob"); ax[2].axis('off')
        ax[3].imshow(cleaned_pred, cmap='gray'); ax[3].set_title(f"Cleaned (T={best_thresh:.2f})"); ax[3].axis('off')
        plt.tight_layout(); plt.show()

if __name__ == "__main__":
    visualize_results()