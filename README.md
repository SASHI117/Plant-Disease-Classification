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

<img width="1804" height="628" alt="Screenshot 2025-11-21 205328" src="https://github.com/user-attachments/assets/2d5e87a6-473a-4070-a28f-7292f09821c6" />
<img width="1801" height="660" alt="Screenshot 2025-11-21 205432" src="https://github.com/user-attachments/assets/923a2d68-6b55-4347-8910-f645be821fc5" />
<img width="1783" height="640" alt="Screenshot 2025-11-21 205926" src="https://github.com/user-attachments/assets/1d4af7bd-8a24-4d1c-966c-68807c8e7345" />
<img width="1788" height="625" alt="Screenshot 2025-11-21 205256" src="https://github.com/user-attachments/assets/c6104d3e-7998-41a1-9de5-7f63b0ce4eb2" />


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
