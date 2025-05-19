# 🏭 Quality Control in Manufacturing Using AI

This project uses computer vision and AI to automate defect detection in manufacturing. The system analyzes product images in real time and identifies visual defects like scratches, spots, or surface anomalies. It helps manufacturers improve quality assurance, reduce manual inspection workload, and maintain consistent product standards.

---

## 📌 Features

- 🔍 **Automated defect detection** using OpenCV and image processing  
- 🖼️ **Reference-based and anomaly-based inspection**  
- 🎯 **Defect classification** (e.g., Scratch, Spot, Dark/Light Areas)  
- 📈 **Visual and textual logging** of defects with bounding boxes  
- 💾 **Automatic saving of annotated results and reports**  
- ⚙️ **Configurable thresholds** and easily extendable  
- 📊 **Scalable to real-time and multi-line environments**  
- 📥 **Data collection and labeling support** for building datasets  

---

## 🔧 Technologies Used

- Python 3.x  
- OpenCV  
- NumPy  
- Matplotlib  
- *(Optional)* Flask – for web dashboard/API  
- *(Optional)* Raspberry Pi / Jetson Nano – for edge deployment  

---

## 📷 How It Works

### 1. Capture Input  
- Loads or captures product images from the manufacturing line.  
- Optionally uses simulated images for testing (`qc_test_images/`).

### 2. Detect Defects  
- Compares each image to a known "good" reference image using image differencing.  
- Alternatively, performs anomaly detection with adaptive thresholding.  

### 3. Classify Defects  
- Based on geometric and pixel intensity features.  
- Types: `Scratch`, `Spot`, `Dark Area`, `Light Area`, `Surface Defect`.

### 4. Save & Report  
- Saves annotated images and generates `.txt` reports with defect information.

### 5. Repeat in Real-Time  
- Can be extended to run continuously in real-time environments.

---

## 📥 Data Collection

This system supports data collection to build a high-quality dataset for:
- Model training (deep learning-based quality control)
- Production audit trails
- Continuous improvement and retraining

### How It Works:
- Captured product images and corresponding defect annotations are saved in:
  - `qc_test_images/` – raw images
  - `qc_results/` – annotated images + `.txt` reports with defect metadata

You can periodically archive these folders or stream the data to cloud storage, databases, or labeling platforms.

---


