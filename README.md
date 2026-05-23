# CNN-PneumoniaMNIST
# Pneumonia Detection: Custom CNN vs. MobileNetV2

[cite_start]This project explores and compares the efficacy of a custom Convolutional Neural Network (CNN) against a pre-trained, fine-tuned MobileNetV2 architecture for the classification of pneumonia using the PneumoniaMNIST dataset[cite: 16].

## Project Overview

[cite_start]The objective was to build and evaluate computer vision models capable of accurately identifying pneumonia from medical imaging data[cite: 2, 178]. [cite_start]The project follows a complete machine learning pipeline, from data preparation and model architecture design to training, evaluation, and optimization[cite: 30, 46, 78, 110, 154].

## Methodology

### 1. Data Preprocessing
* [cite_start]**Custom CNN:** Images were resized to 28x28, converted to tensors, and normalized[cite: 27, 33, 34].
* [cite_start]**MobileNetV2:** Given its architecture requirements, images were resized to 224x224, converted to 3-channel grayscale, and augmented using random horizontal flips and rotations to improve robustness[cite: 28, 49, 50, 51, 52].

### 2. Model Architectures
* [cite_start]**Custom CNN:** A sequential model consisting of three convolutional layers with Batch Normalization, ReLU activation, Max Pooling, and a final fully connected classifier[cite: 81, 83, 84, 85, 86, 97].
* [cite_start]**MobileNetV2:** Utilized a pre-trained MobileNetV2 base with frozen parameters, fine-tuning only the final three blocks and the classifier layer to adapt to the specific medical task[cite: 111, 113, 115, 117, 118].

### 3. Training & Optimization
* [cite_start]Both models were trained using `BCEWithLogitsLoss`[cite: 154, 173].
* [cite_start]An Adam optimizer was implemented with a `ReduceLROnPlateau` scheduler to dynamically adjust the learning rate based on validation loss[cite: 171, 181].
* [cite_start]Training involved monitoring loss, accuracy, and Area Under the Curve (AUC)[cite: 158, 161].

## Results

After training, the models were evaluated on the test set. [cite_start]Threshold adjustment was performed on the custom CNN to optimize the balance between sensitivity and specificity[cite: 275, 278].

| Model | Test Accuracy | Test AUC |
| :--- | :--- | :--- |
| Custom CNN (Initial) | [cite_start]0.8446 [cite: 180] | [cite_start]0.9668 [cite: 180] |
| MobileNetV2 | [cite_start]0.8221 [cite: 186] | [cite_start]0.9581 [cite: 186] |
| Custom CNN (Optimized) | [cite_start]0.9100 [cite: 280] | N/A |

### Optimized Performance (Custom CNN)
[cite_start]By adjusting the classification threshold to 0.999 based on the ROC curve, the custom CNN achieved a significant performance boost[cite: 275, 276, 280]:
* [cite_start]**Accuracy:** 0.91 [cite: 280]
* [cite_start]**Precision (Pneumonia):** 0.93 [cite: 280]
* [cite_start]**Recall (Pneumonia):** 0.92 [cite: 280]

## Tools and Libraries
* [cite_start]**Framework:** PyTorch [cite: 11]
* [cite_start]**Dataset:** MedMNIST (PneumoniaMNIST) [cite: 16]
* [cite_start]**Evaluation:** Scikit-learn [cite: 18]
* [cite_start]**Visualization:** Seaborn, Matplotlib [cite: 17, 19]

---
[cite_start]*Developed as a computer vision exploration using Google Colab[cite: 2, 41].*
