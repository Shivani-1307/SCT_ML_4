# SCT_ML_4 - Real-Time Hand Gesture Recognition
Developed a real-time Hand Gesture Recognition system using Python, TensorFlow, and OpenCV. Trained a custom CNN on the LeapGestRecog dataset to classify 10 distinct gestures. Implemented a robust computer vision pipeline utilizing YCrCb skin color segmentation to eliminate background noise and ensure fluid, touchless human-computer interaction.

# Real-Time Hand Gesture Recognition (ML/DL Pipeline)

A robust Deep Learning system that classifies 10 distinct hand gestures in real-time from video streams. This project covers the full Machine Learning lifecycle: from data preprocessing and augmentation to model architecture design and real-time live inference using computer vision techniques.

## 🧠 Machine Learning Overview

* **Task Type:** Multi-class Image Classification (10 Classes)
* **Dataset:** LeapGestRecog (Infrared grayscale hand gesture images)
* **Core Model Architecture:** Sequential Convolutional Neural Network (CNN)
* **Key ML Challenges Addressed:** Overfitting (mitigated via Data Augmentation and Dropout) and Domain Adaptation (matching noisy RGB webcam feeds to clean infrared dataset styles via Skin Color Segmentation).

---

## 🛠️ System Architecture & ML Pipeline

```text
[Input Video Frame] ──> [Crop Region of Interest] ──> [YCrCb Skin Segmentation]
                                                               │
[Trained CNN Model] <── [Resize & Normalize] <── [Binary Mask Silhouette]
        │
[Softmax Output] ──> [Gesture Classification & Action Trigger]

