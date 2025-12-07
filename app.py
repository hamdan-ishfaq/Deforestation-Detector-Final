import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import tifffile as tiff
from PIL import Image
import os
import sys

# --- CRITICAL: Ensure model_custom6.py is in the root directory ---
from model_custom6 import AttentionResUNet 

# --- 1. CONFIGURATION ---
CHECKPOINT_PATH = "model_final.pth" 
DEVICE = "cpu" # Must be CPU for Streamlit Cloud
OPTIMAL_THRESHOLD = 0.90 # Use your found optimal threshold

@st.cache_resource
def load_model():
    """Loads and caches the model, ensuring CPU mapping."""
    try:
        model = AttentionResUNet(in_channels=9, out_channels=2).to(DEVICE)
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# --- 2. PREPROCESSING LOGIC ---
def preprocess_tif_to_tensor(tif_file_stream):
    """
    Reads the 9-band TIFF, normalizes, and prepares the tensor (C, H, W).
    """
    try:
        X = tiff.imread(tif_file_stream).astype(np.float32)
    except Exception as e:
        st.error(f"Error reading TIFF file: {e}")
        st.stop()
        
    # Ensure shape is (C, H, W). GeoTIFFs often use (H, W, C).
    if X.ndim == 3 and X.shape[2] == 9:
        X = np.transpose(X, (2, 0, 1))
    
    if X.shape[0] != 9:
        st.error(f"Expected 9 input channels (Bands), but found {X.shape[0]}.")
        st.stop()
        
    # Apply normalization (Assumes data is raw DN values and needs scaling)
    X = np.nan_to_num(X, nan=0.0)
    # Simple normalization based on max value to scale to [0, 1]
    max_val = np.max(X)
    X = np.clip(X / max_val, 0.0, 1.0) if max_val > 1.0 else np.clip(X, 0.0, 1.0)
    
    X_tensor = torch.tensor(X).unsqueeze(0).float().to(DEVICE)
    return X_tensor

def postprocess_and_display(mask_tensor):
    """Applies threshold and calculates area."""
    
    mask_array = mask_tensor.cpu().numpy()
    mask_bin = (mask_array > OPTIMAL_THRESHOLD).astype(np.uint8)
    
    deforested_pixels = np.sum(mask_bin)
    deforestation_percentage = (deforested_pixels / mask_bin.size) * 100
    
    mask_img_visual = Image.fromarray((mask_bin * 255).astype(np.uint8))
    
    return mask_img_visual, deforestation_percentage

# --- 3. STREAMLIT UI ---
st.set_page_config(page_title="Deforestation Detector", layout="wide")
st.title("🌳 Final Deforestation Detector")

model = load_model()

uploaded_file = st.file_uploader("Upload 9-Band GeoTIFF Patch (256x256)", type=["tif", "tiff"])

if uploaded_file and model:
    
    X_tensor_9ch = preprocess_tif_to_tensor(uploaded_file)
    
    with st.spinner('Running Model Prediction...'):
        with torch.no_grad():
            seg_logits, _, global_pred = model(X_tensor_9ch)
            pred_prob_tensor = F.softmax(seg_logits, dim=1)[0, 1]
    
    mask_visual, area_percent = postprocess_and_display(pred_prob_tensor)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Predicted Deforestation Mask")
        st.image(mask_visual, caption=f"Thresholded at {OPTIMAL_THRESHOLD}", use_column_width=True)
        
    with col2:
        st.subheader("Analysis Summary")
        # Global prediction head returns a single value (e.g., probability of deforestation in the patch)
        st.metric(label="Global Deforestation Confidence", value=f"{float(global_pred[0]):.2f}")
        st.metric(label="Total Deforested Area Coverage", value=f"{area_percent:.2f}%")