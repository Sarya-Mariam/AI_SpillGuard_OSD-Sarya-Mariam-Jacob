# 🌊 AI SpillGuard – Oil Spill Detection & Monitoring System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange.svg" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Models-U--Net|Dual--Head--U--Net|DeepLabV3+-green.svg" alt="Models">
  <img src="https://img.shields.io/badge/Status-Complete-success.svg" alt="Status">
</p>

---

## 📌 Project Overview

**AI SpillGuard** is a deep learning-based oil spill detection and monitoring system that uses **Sentinel-1 SAR satellite imagery** to automatically identify and segment oil-contaminated regions in the ocean.

The system was developed using three different deep learning models — **U-Net**, **Dual-Head U-Net**, and **DeepLabV3+** — to determine the most effective architecture for precise oil spill segmentation.

By automating detection, the system enables **faster response**, **accurate monitoring**, and **reduced environmental damage** compared to manual or traditional methods.

---

## 🎯 Objectives

* Detect and segment oil spills from satellite images automatically
* Compare performance of U-Net, Dual-Head U-Net, and DeepLabV3+
* Classify spill vs. non-spill regions
* Generate clear visual overlays for interpretation
* Provide a deployable real-time monitoring interface

---

## 🚀 Key Features

✅ **Multi-Model Training** – Implemented and compared U-Net, Dual-Head U-Net, and DeepLabV3+
✅ **SAR-Specific Preprocessing** – Includes normalization and speckle noise reduction
✅ **Advanced Augmentation** – Flipping, rotation, brightness and contrast variations
✅ **Hybrid Loss Function** – Binary Cross-Entropy + Dice Loss
✅ **Comprehensive Evaluation Metrics** – IoU, Dice, Accuracy, Precision, Recall
✅ **Visual Analytics** – Overlay segmentation, heatmaps, and side-by-side comparison
✅ **Streamlit Deployment** – Real-time inference and result visualization

---

## 🏗️ System Architecture

### Workflow

```
Satellite Image Input
        ↓
Data Preprocessing & Augmentation
        ↓
Model Training (U-Net / Dual-Head U-Net / DeepLabV3+)
        ↓
Segmentation & Classification Output
        ↓
Evaluation & Visualization
        ↓
Streamlit Deployment Interface
```

### Technology Stack

* **Deep Learning Framework:** TensorFlow / Keras
* **Processing Libraries:** OpenCV, PIL, NumPy
* **Visualization Tools:** Matplotlib, Seaborn
* **Deployment:** Streamlit
* **Dataset Source:** [Zenodo Oil Spill Dataset (Sentinel-1 SAR)](https://zenodo.org/records/10555314/files/dataset.zip?download=1)

---

## 📊 Results & Model Comparison

| Model               | Dice Coefficient | IoU      | Accuracy  | Precision | Recall    |
| ------------------- | ---------------- | -------- | --------- | --------- | --------- |
| **U-Net**           | 0.71             | 0.85     | 91.8%     | 85.6%     | 82.3%     |
| **Dual-Head U-Net** | 0.73             | 0.87     | 92.4%     | 87.2%     | 84.6%     |
| **DeepLabV3+**      | **0.78**         | **0.90** | **94.1%** | **89.5%** | **87.9%** |

🧠 **DeepLabV3+** demonstrated the highest segmentation performance, producing more accurate and smoother spill boundaries compared to U-Net variants.

---

## 📋 Implementation Milestones (Summarized)

### 🧩 Milestone 1: Data Preparation

* Collected Sentinel-1 SAR dataset from **Zenodo**
* Resized images to 256×256 and normalized pixel values
* Reduced SAR-specific speckle noise
* Performed data augmentation (flip, rotate, adjust brightness/contrast)
* Split into training, validation, and testing sets

### 🧠 Milestone 2: Model Development

* **U-Net:** Baseline segmentation model
* **Dual-Head U-Net:** Added classification head for spill detection
* **DeepLabV3+:** Advanced encoder-decoder with atrous convolutions
* Trained all models using **Adam optimizer** and **BCE + Dice Loss**
* Integrated callbacks: EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau

### 📈 Milestone 3: Evaluation & Visualization

* Computed **Accuracy, IoU, Dice, Precision, Recall**
* Visualized results with overlays, confusion matrices, and heatmaps
* Compared model performance across all three architectures

### 🌐 Milestone 4: Deployment

* Developed a **Streamlit web interface** for real-time inference
* Enabled drag-and-drop image uploads and visual result display
* Added downloadable mask outputs for further analysis

---

## 🧠 Technical Summary

### Model Architectures

* **U-Net:** Classic encoder-decoder design with skip connections.
* **Dual-Head U-Net:** Adds a classification head for global prediction (spill/non-spill).
* **DeepLabV3+:** Uses atrous spatial pyramid pooling (ASPP) for multi-scale context and superior segmentation accuracy.

### Loss Function

**Total Loss = Binary Cross-Entropy + Dice Loss**
Balances pixel-wise classification with region overlap optimization.

---

## 📈 Future Enhancements

* Multi-class severity segmentation (light, moderate, severe)
* Integration with live Sentinel-1 satellite feeds
* Real-time alert system for spill detection
* Edge/mobile deployment for field operations
* Temporal analysis for spill progression tracking

---

<p align="center">
  🌊 <b>AI SpillGuard</b> – Harnessing AI to protect our oceans 🌊  
  <br><i>“Detect early. Act fast. Preserve marine life.”</i>
</p>

---
