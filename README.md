# 🧠 NeuroScan AI – DenseNet-Based Brain Tumor Classification Desktop App

A Deep Learning–powered medical imaging analyzer with Grad-CAM explainability

---

## 📌 Overview

**NeuroScan AI** is an advanced desktop application built with **Tkinter (Python GUI)** and a custom-trained **DenseNet-121** model to classify brain MRI scans into four categories:

- **Glioma**
- **Healthy**
- **Meningioma**
- **Pituitary**

The system provides fast, reliable, and explainable predictions using:

✔️ Deep Learning  
✔️ Grad-CAM visualization  
✔️ Softmax probability distribution  
✔️ Interactive UI  

The app displays both the original MRI and the AI-generated attention map for explainability.

---

## 🎯 Project Goal

This project aims to:

- Assist in early diagnosis of brain tumors  
- Demonstrate explainable AI via Grad-CAM  
- Provide user-friendly visualization  
- Serve as an academic / research-grade tool  
- Act as a base for future models (ResNet, EfficientNet, Ensembles)

---

## 🚀 Features


### 🧠 Deep Learning Model
- DenseNet-121 pretrained on ImageNet  
- Modified classifier for 4-class tumor detection  
- Added dropout for better generalization  
- CUDA GPU support  

### 🔍 Explainability
- Integrated **Grad-CAM** heatmaps  
- Highlights model attention regions  
- Overlays heatmap on MRI image  

### 📊 Prediction Dashboard
Displays:
- Predicted class  
- Confidence score  
- Probability bars for all four classes  

### 📁 Image Loader
- Accepts **PNG, JPG, JPEG, TIFF, BMP**  
- Auto-resizes and centers inside UI cards  

### ⚡ Fast Inference
- One-click analysis  
- Optimized preprocessing  
- Minimal latency  

---

## 🧱 Tech Stack

| Area                   | Technology            |
|------------------------|------------------------|
| Deep Learning Framework | PyTorch               |
| Model Architecture      | DenseNet-121          |
| Explainability          | Grad-CAM (pytorch-grad-cam) |
| GUI Framework           | Tkinter               |
| Image Processing        | Pillow (PIL)          |
| Visualization           | ImageDraw, ImageTk    |
| Hardware Support        | CPU / CUDA GPU        |

---

## Project Structure

NeuroScan-AI/
│── app.py                     # Main GUI application
│── dense_model.pth            # Trained DenseNet model
│── README.md                  # Documentation
│── requirements.txt
│── DenseNet.ipynb


---

## 🧪 Model Details
Training Setup

- Architecture: DenseNet-121

- Pretraining: ImageNet

- Loss: CrossEntropyLoss

- Optimizer: Adam / SGD


## Classes

- Glioma

- Healthy

- Meningioma

- Pituitary


---

## Team

| Name                    | USN                                      |
| ----------------------- | ----------------------------------------- |
| **Bhuvan S**            | 4MH22CA009                                |
| **Thilak R**            | 4MH22CA057                                |
| **Prajwal Koundinya P** | 4MH22CA031                                |



