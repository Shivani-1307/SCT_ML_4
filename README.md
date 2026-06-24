# SCT_ML_4 - Real-Time Hand Gesture Recognition

A real-time Hand Gesture Recognition system developed using Python, TensorFlow, and OpenCV. The project uses a Convolutional Neural Network (CNN) trained on the LeapGestRecog dataset to classify 10 distinct hand gestures. The system performs live webcam inference using a skin color segmentation pipeline in the YCrCb color space, enabling touchless human-computer interaction.

## 🚀 Features

* Real-time hand gesture recognition using webcam
* CNN-based deep learning model
* Classification of 10 different hand gestures
* YCrCb skin color segmentation for hand detection
* Background noise reduction using morphological operations
* Live confidence score display
* Gesture recognition from video streams

---

## 📊 Dataset

**Dataset:** LeapGestRecog

The model is trained using the LeapGestRecog dataset containing grayscale infrared images of hand gestures.

Download Dataset:
https://www.kaggle.com/datasets/gti-upm/leapgestrecog

After downloading, extract the dataset into:

```text
leapGestRecog/
├── 00/
├── 01/
├── 02/
...
├── 09/
```

---

## 🧠 Machine Learning Overview

* Task Type: Multi-Class Image Classification
* Number of Classes: 10
* Model: Convolutional Neural Network (CNN)
* Framework: TensorFlow / Keras
* Image Size: 64 × 64
* Input Type: Grayscale Images
* Output Layer: Softmax (10 Classes)

### Techniques Used

* Data Augmentation

  * Rotation
  * Zoom
  * Width Shift
  * Height Shift

* Regularization

  * Dropout Layer (0.5)

* Image Processing

  * YCrCb Skin Color Segmentation
  * Morphological Opening
  * Gaussian Blur

---

## 🛠️ System Architecture

```text
Webcam Frame
      │
      ▼
Region of Interest (ROI)
      │
      ▼
YCrCb Skin Segmentation
      │
      ▼
Morphological Filtering
      │
      ▼
Resize (64×64)
      │
      ▼
Normalization
      │
      ▼
CNN Model
      │
      ▼
Softmax Classification
      │
      ▼
Gesture Prediction
```

---

## ✋ Supported Gestures

| Label | Gesture    |
| ----- | ---------- |
| 01    | Palm       |
| 02    | L          |
| 03    | Fist       |
| 04    | Fist Moved |
| 05    | Thumb      |
| 06    | Index      |
| 07    | OK         |
| 08    | Palm Moved |
| 09    | C          |
| 10    | Down       |

---

## 📂 Project Structure

```text
SCT_ML_4/
│
├── train.py
├── test_webcam.py
├── hand_gesture_model.h5
├── requirements.txt
├── README.md
└── leapGestRecog/
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/SCT_ML_4.git
cd SCT_ML_4
```

Install dependencies:

```bash
pip install tensorflow
pip install opencv-python
pip install numpy
pip install scikit-learn
```

---

## 🎯 Training the Model

Run:

```bash
python train.py
```

The trained model will be saved as:

```text
hand_gesture_model.h5
```

---

## 📷 Real-Time Gesture Recognition

Run:

```bash
python test_webcam.py
```

Controls:

* Press **Q** to quit.
* Place your hand inside the green bounding box.
* The system will display the detected gesture and confidence score.

---

## 📸 Output

### Figure 1: Live Gesture Recognition

The webcam captures hand gestures in real time. A Region of Interest (ROI) is extracted and processed using YCrCb skin color segmentation. The segmented hand image is passed to the trained CNN model, which predicts the gesture class and displays the confidence score on the screen.

### Figure 2: Skin Segmentation View

The segmentation window displays the binary hand mask generated after applying skin color filtering and morphological operations. This preprocessing step helps reduce background noise and improve recognition accuracy.

---

## 📈 Results

* High classification accuracy on LeapGestRecog dataset
* Real-time prediction using webcam input
* Robust hand extraction using skin color segmentation
* Smooth gesture classification with confidence scoring

---

## 👨‍💻 Author

Shivani Banuka


