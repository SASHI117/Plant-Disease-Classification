# 🌱 Plant Disease Classification using MobileNetV2

A deep-learning–based system for classifying plant leaf diseases using the **PlantVillage** dataset.  
This project uses **MobileNetV2**, **Transfer Learning**, and a **Gradio web interface** to deliver fast and accurate predictions across **15 disease classes** from 20,000+ images.

---

## 📌 Project Overview

This project aims to help farmers and researchers quickly detect plant diseases from leaf images.  
Using MobileNetV2 and fine-tuning techniques, the model achieves **94%+ validation accuracy** while remaining lightweight and deployable on web or mobile.

---

## 🚀 Features

- ✔ **MobileNetV2-based deep learning model**
- ✔ **Trained on 20k+ PlantVillage images (15 classes)**
- ✔ **Two-phase training (head training + fine-tuning)**
- ✔ **Real-time prediction using Gradio**
- ✔ **.h5 model saved for deployment**
- ✔ Works on CPU (Google Colab) without GPU
- ✔ Clean, modular, reproducible code

---

## 📁 Dataset

- **Source:** PlantVillage (Kaggle)
- **Images:** ~20,000
- **Classes:** 15
- **Split:** 80% training, 20% validation
- **Format:** RGB leaf images organized by folder structure

---

## 🧠 Model Architecture

- **Base Model:** MobileNetV2 (ImageNet weights)
- **Frozen layers:** All except last 40 during fine-tuning
- **Added Layers:**
  - GlobalAveragePooling2D
  - Dense(256, ReLU)
  - Dropout(0.3)
  - Dense(15, Softmax)

---

## 🔧 Training Details

### **Phase 1: Train Classifier Head**
- Epochs: 3  
- LR: 1e-3  

### **Phase 2: Fine-tune Last 40 Layers**
- Epochs: 4  
- LR: 1e-4  

### **Final Results**
- **Training Accuracy:** 95.6%
- **Validation Accuracy:** 94.3%
- **Final Loss:** 0.17

---

## 🖥️ Gradio Web App

A simple Gradio interface is included for:

- Uploading leaf images  
- Displaying top-3 predicted diseases  
- Showing confidence scores  

Run locally:

```python
import gradio as gr
interface.launch()
```
Here is the exact section you asked for, rewritten cleanly and formatted in README markdown hash (#) format and wrapped inside a code block so you can paste it directly into GitHub.

## ▶️ How to Run

### **1. Install dependencies**
```bash
pip install tensorflow keras gradio numpy
```
### **2. Run the Gradio app**
```bash
python app.py
```
### **4. Upload a leaf image**

Once the app opens in your browser, upload a plant leaf image.

The model will return:

Predicted class

Confidence score

### **📌 Example Predictions**

![Prediction Example](C:\Users\mesas\OneDrive\Pictures\Screenshots\Screenshot 2025-11-21 205328.png)

### **🛠️ Future Improvements**

Convert the model to TensorFlow Lite for mobile deployment

Add Grad-CAM heatmaps for model explainability

Implement multi-image batch prediction

Deploy a live version on Hugging Face Spaces

Add treatment suggestions for each detected disease

### **📄 License**

This project is licensed under the MIT License.

### **🙌 Acknowledgements**

PlantVillage Dataset (Kaggle)

MobileNetV2 Architecture (Google Research)

TensorFlow/Keras

Gradio Interface
