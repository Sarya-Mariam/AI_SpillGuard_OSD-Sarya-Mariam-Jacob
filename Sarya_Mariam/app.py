import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import gdown
import os
import zipfile

GOOGLE_DRIVE_ID = "1k-5vuKHInd1ClXz2Mql8Z_UGjtbYbAxg"
MODEL_PATH = None  

# Download and extract model if not already present
if not any(f.endswith(".h5") for f in os.listdir(".")):
    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}"
    gdown.download(url, "dual_head_model.zip", quiet=False)

    with zipfile.ZipFile("dual_head_model.zip", 'r') as zip_ref:
        zip_ref.extractall(".")

# Find the first .h5 file
for file in os.listdir("."):
    if file.endswith(".h5"):
        MODEL_PATH = file
        break

if MODEL_PATH is None:
    raise FileNotFoundError("No .h5 model file found after extraction!")

# Load model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

IMG_SIZE = 256

def preprocess_image(img: Image.Image):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    return arr

st.title("🌊 AI SpillGuard – Oil Spill Detection")
st.write("Upload a satellite image to detect oil spill regions using a trained Dual Head U-Net model.")

# --- 🔧 Adjustable parameters ---
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.6, 0.05)
area_thresh = st.sidebar.slider("Spill Area Threshold (%)", 0.0, 20.0, 5.0, 0.5)

uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg","jpeg","png","tif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    arr = preprocess_image(image)
    pred = model.predict(np.expand_dims(arr, 0))[0]

    # Apply threshold from slider
    pred_bin = (pred[:,:,0] > conf_thresh).astype("uint8")

    # Morphological filtering
    kernel = np.ones((3,3), np.uint8)
    pred_bin = cv2.morphologyEx(pred_bin, cv2.MORPH_OPEN, kernel)
    pred_bin = cv2.morphologyEx(pred_bin, cv2.MORPH_CLOSE, kernel)

    # Compute spill ratio
    spill_ratio = np.sum(pred_bin) / pred_bin.size

    if spill_ratio > (area_thresh / 100.0):
        st.success(f"🌊 Oil Spill Detected! (~{spill_ratio*100:.2f}% of image)")
    else:
        st.info("✅ No Oil Spill Detected")

    # Debug heatmap
    st.subheader("Model Output Heatmap")
    heatmap = (pred[:,:,0] * 255).astype("uint8")
    st.image(heatmap, caption="Raw Prediction Probabilities", use_container_width=True, channels="GRAY")

    # Overlay visualization
    img_bgr = cv2.cvtColor(
        np.array(image.resize((IMG_SIZE, IMG_SIZE)), dtype=np.uint8),
        cv2.COLOR_RGB2BGR
    )
    mask = (pred_bin * 255).astype("uint8")
    mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

    if img_bgr.shape != mask_color.shape:
        mask_color = cv2.resize(mask_color, (img_bgr.shape[1], img_bgr.shape[0]))

    overlay = cv2.addWeighted(img_bgr, 0.7, mask_color, 0.3, 0)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    st.image(overlay, caption="Predicted Oil Spill Regions", use_container_width=True)































