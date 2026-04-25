# AI-Based Human Activity Recognition (CNN-LSTM)

## 📌 Description

This project implements a real-time Human Activity Recognition (HAR) system using a CNN-LSTM deep learning model. It supports both webcam-based detection and video file processing, and can recognize activities such as walking, sitting, clapping, running, and more.

---

## 🚀 Features

* Real-time webcam activity detection
* Video file-based detection
* Activity classification with confidence score
* Pre-trained model (no training required)

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy

---

## 🧠 Model Architecture

- CNN layers for spatial feature extraction
- LSTM layer for temporal sequence learning
- Fully connected layers for classification
- Output layer with Softmax activation (18 classes)

---

## 📂 Project Files

* main.py → Main program
* har_model_tf.zip → Trained model
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

The system provides the following outputs:

* Detects human activity in real-time (webcam) or from video input  
* Draws a bounding box around the detected person  
* Displays the predicted activity label on the screen  
* Shows confidence score (accuracy percentage) for each prediction  
* Continuously updates predictions based on live input  

---

## 📊 Dataset

The dataset used in this project is a combination of multiple publicly available Human Activity Recognition datasets.

A total of **18 activity classes** are included, such as walking, running, clapping, sitting, jumping, etc.

📥 Dataset Download:
https://drive.google.com/file/d/1WYHajUtvfQIopIRAg3aPuLWMol6BHq2V/view?usp=drivesdk

Note: The dataset is not included in this repository due to size constraints.

---

## ⚠️ Notes

* Ensure webcam is connected for live detection
* The model is already trained and ready to use
* Keep all files in the same folder
* Do not rename required files

---

## 👩‍💻 Author
REENA S
