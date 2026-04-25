# AI-Based Human Activity Recognition

## 📌 Description

This project detects human activities using a CNN-LSTM deep learning model.
It supports both real-time webcam input and video file processing.

---

## 🚀 Features

* Real-time webcam activity detection
* Video file-based detection
* Activity classification with confidence score
* Pre-trained model (no training required)

---

## 📂 Project Files

* main.py → Main program
* har_model.keras → Trained model
* labels.pkl → Class labels
* model_config.json → Configuration file
* haarcascade XML files → Human detection files
* requirements.txt → Required Python libraries

---

## ⚙️ Requirements

* Python 3.8 or above

Install dependencies using:

pip install -r requirements.txt

---

## ▶️ How to Run

1. Open terminal in the project folder

2. Run the program:

   python main.py

3. Select mode:

   * 1 → Webcam Detection
   * 2 → Video File Detection

4. For video mode:

   * Place the video file inside the project folder
   * Enter only the file name (no quotes)

   Example:
   sample.mp4

---

## 📊 Output

* Displays detected human activity
* Shows predicted label and confidence score on screen

---

## ⚠️ Notes

* Ensure webcam is connected for live detection
* The model is already trained and ready to use
* Keep all files in the same folder
* Do not rename required files
