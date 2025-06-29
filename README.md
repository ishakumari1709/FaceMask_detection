
# Face Mask Detector

This is a lightweight yet effective **Face Mask Detector** built using **Python**, **TensorFlow/Keras**, and **Streamlit**. It classifies face images into two categories: **Mask** or **Non-Mask**. The model is trained from scratch using a custom CNN and deployed locally via a web interface.

---

## 🔍 Project Overview

Face masks have become essential for public health. This tool enables quick classification of face images to detect compliance using a minimal CNN — suitable for fast training and offline usage.

---
| Library         | Purpose                                |
|----------------|----------------------------------------|
| `tensorflow`    | Model building and training (Keras API)|
| `opencv-python` | Image preprocessing for test script    |
| `pillow`        | Image handling in Streamlit            |
| `streamlit`     | Lightweight local web interface        |

---

## ✅ Features

- Train a custom CNN on mask/non-mask images
- Upload `.jpg` or `.png` face images for prediction
- Simple Streamlit app to view results
- Shows label (`😷 Mask` / `🚫 Non Mask`) and confidence
- Progress bar and editable layout
- Supports local images for testing via CLI

---

## 🧠 Techniques Used

### 🎯 Convolutional Neural Network (CNN)
- Detects spatial features like mouth/nose covering
- Two convolution + pooling layers for speed & simplicity

### 🧰 Keras `.keras` model format
- Modern and robust
- Replaces older `.h5` format

### 🌐 Streamlit Web Interface
- Upload & predict from browser
- Quick demo UI for non-technical users

---

## 🧪 How It Works

1. Dataset is split into:
   - `Mask/`
   - `Non_Mask/`
2. CNN model is trained for binary classification
3. Model is saved as `mask_model_small.keras`
4. Predictions:
   - Via command-line: `test.py`
   - Via UI: `streamlit run app.py`
5. Confidence is shown along with label

---

## 📦 Folder Structure

<pre>
FaceMaskDetector/
├── data/
│   ├── Mask/
│   └── Non_Mask/
├── train.py             # CNN training script
├── test.py              # CLI-based image tester
├── app.py               # Streamlit web app
├── mask_model_small.keras  # Saved model
├── requirements.txt     # Dependencies
└── README.md
</pre>

---

## 📸 Screenshots

![Screenshot 2025-06-29 185629](https://github.com/user-attachments/assets/f04db668-7a98-4b3f-8201-d33e44d2fb06)
![Screenshot 2025-06-29 185536](https://github.com/user-attachments/assets/832a90a7-cf82-479c-82a4-0a7593a66d98)


---

## ▶️ Short Demo
Coming soon (record using Loom, OBS, or ScreenRec)

---

## ⚙️ Setup Guide

### 1. Clone the repository
```bash
git clone https://github.com/your-username/face-mask-detector.git
cd face-mask-detector
```

### 2. (Optional) Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate     # On Windows
source .venv/bin/activate  # On macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
```bash
python train.py
```

### 5. Test on a single image
```bash
python test.py path/to/image.jpg
```

### 6. Launch Streamlit app
```bash
streamlit run app.py
```

---

## 🔧 Customization

- Add webcam support (OpenCV)
- Improve model with more layers or MobileNetV2
- Add Grad-CAM for visual explanation
- Deploy using Hugging Face Spaces or Streamlit Cloud

---



---
