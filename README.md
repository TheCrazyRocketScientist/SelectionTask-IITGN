# SelectionTask-IITGN
A repo for the second selection task for SRIP 2025(Project Number: IP0NB0000020))

# **Vibration Touch Panel Detection Using Machine Learning**  

[![Project Demo](link-to-demo-gif)](link-to-full-video)  <!-- Add this if possible -->

## **Overview**
This project explores **tap detection on a glass panel** using an accelerometer (ADXL345) and an ESP32 microcontroller. Machine learning models (RandomForest, LightGBM, XGBoost) were used to classify vibrations, enabling precise touch interaction.

**Key Highlights:**
- Real-time **wireless data transmission** using WebSockets  
- **Feature extraction** from raw acceleration data using a **sliding window approach**  
- **Supervised learning conversion** of time-series data  

## **Hardware & Setup**
### **Hardware Used**
- **ADXL345** accelerometer (for tap detection)  
- **ESP32 WROOM32** (for data processing & wireless communication)  
- **LED indicators** (for tap classification)  

![Hardware Setup](link-to-image)  <!-- Add a clear image -->

### **Circuit Diagram**
*(Include a basic Fritzing diagram if possible.)*

---

## **Software Implementation**
### **1️⃣ Data Collection**
- Raw acceleration data `[X, Y, Z, Tap, Double_Tap]` collected at **100Hz**  
- Data stored as a **CSV file** for training (imu_data.csv in the repo) 

### **2️⃣ Feature Engineering**
- Sliding window approach: **Window size: 50 samples (0.5 sec), Overlap: 25 samples (50%)**  
- Extracted features:
  - **Statistical Features of Time and Frequency-domain**: Mean, Max, Min, Variance, IQR, etc.
  - **Frequency-domain exclusive (Future Work)**: energy, entropy, SMA 

### **3️⃣ Model Training**
- **Models Used:** RandomForest, XGBoost, LightGBM  
- **Feature Selection:** Recursive Feature Elimination (RFE) → Selected **25 best features**  
- **Hyperparameter tuning:** Grid Search CV 

### **4️⃣ Real-Time Inference**
- Buffer-based real-time prediction  
- Extracts features from **live ESP32 sensor data**  
- **Predicted labels sent back to ESP32**
---

## **Results & Performance**
| Model | Accuracy | Precision | Recall |
|--------|----------|-----------|--------|
| RandomForest | **95.2%** | 93.4% | 94.1% |
| XGBoost | **96.1%** | 94.7% | 95.3% |
| LightGBM | **94.8%** | 92.9% | 93.8% |

 **Observations:**
- XGBoost performed the best, likely due to **handling tabular data better**  
- **Feature selection improved model stability**  

---

## **Future Work**
 **Multi-region tap detection**: Dividing the glass into sections for finer interaction  
**Testing hardware independence**: Swapping ADXL345 with MPU6050  

---

## **Setup & Usage**
### **🔹 1. Install Dependencies**
```sh
pip install -r requirements.txt
