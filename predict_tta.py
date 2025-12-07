import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import random

from model_custom6 import AttentionResUNet
from dataset import PatchDataset

# --- CONFIGURATION ---
CHECKPOINT_PATH = "checkpoints/attention_softfp_model.pth"
DATA_DIR = Path("data/patches_npz")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OPTIMAL_THRESHOLD = 0.90  # The magic number you found
MIN_AREA = 50             # Remove blobs smaller than 50 pixels

def clean_mask(mask_prob, threshold, min_object_size):
    # 1. Hard Threshold
    mask_bin = (mask_prob > threshold).astype(np.uint8)
    
    # 2. Morphological Opening (Scrubbing tiny noise)
    kernel = np.ones((3,3), np.uint8)
    mask_clean = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 3. Area Filtering (Removing small islands)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_object_size:
            mask_clean[labels == i] = 0
            
    return mask_clean

def predict_with_tta(model, input_tensor):
    """
    Test-Time Augmentation (TTA).
    Predicts on the image, and horizontal/vertical flips, then averages.
    This smooths out noise and boosts confidence.
    """
    model.eval()
    with torch.no_grad():
        # 1. Original
        logits, _, _ = model(input_tensor)
        p1 = F.softmax(logits, dim=1)[:, 1, :, :]
        
        # 2. Horizontal Flip
        x_h = torch.flip(input_tensor, [3])
        logits_h, _, _ = model(x_h)
        p2 = F.softmax(logits_h, dim=1)[:, 1, :, :]
        p2 = torch.flip(p2, [2]) # Flip back (B, H, W) -> dim 2 is width after removing channels
        
        # 3. Vertical Flip
        x_v = torch.flip(input_tensor, [2])
        logits_v, _, _ = model(x_v)
        p3 = F.softmax(logits_v, dim=1)[:, 1, :, :]
        p3 = torch.flip(p3, [1]) # Flip back
        
        # Average the predictions
        p_avg = (p1 + p2 + p3) / 3.0
        
    return p_avg.cpu().numpy()[0] # Return (H, W) array

def run_final_visualization(num_samples=5):
    dataset = PatchDataset(DATA_DIR)
    
    print(f"Loading model... Threshold set to {OPTIMAL_THRESHOLD}")
    model = AttentionResUNet(in_channels=9, out_channels=2).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    # Filter for interesting images
    viz_indices = []
    for idx in indices:
        _, mask = dataset[idx]
        if mask.sum() > 50: 
            viz_indices.append(idx)
        if len(viz_indices) >= num_samples:
            break
            
    for idx in viz_indices:
        image, mask = dataset[idx] 
        input_tensor = image.unsqueeze(0).to(DEVICE)
        
        # --- PREDICT WITH TTA ---
        pred_prob = predict_with_tta(model, input_tensor)
        
        # --- CLEAN ---
        final_mask = clean_mask(pred_prob, OPTIMAL_THRESHOLD, MIN_AREA)
        
        # --- PLOT ---
        gt_mask = mask.squeeze().numpy()
        input_img_viz = image[3].numpy() # Red Band
        
        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        ax[0].imshow(input_img_viz, cmap='gray')
        ax[0].set_title("Input (Red Band)")
        ax[0].axis('off')
        
        ax[1].imshow(gt_mask, cmap='gray')
        ax[1].set_title("Ground Truth")
        ax[1].axis('off')
        
        ax[2].imshow(pred_prob, cmap='jet', vmin=0, vmax=1)
        ax[2].set_title("AI Confidence (TTA)")
        ax[2].axis('off')
        
        ax[3].imshow(final_mask, cmap='gray')
        ax[3].set_title(f"Final Output (T={OPTIMAL_THRESHOLD})")
        ax[3].axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    run_final_visualization()