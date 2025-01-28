# Healthcare Management System

## Overview
This repository contains a comprehensive healthcare management system that includes various applications for yoga pose tracking, disease prediction, mental health analysis, and more. The system leverages machine learning models, computer vision, and natural language processing to provide real-time feedback and insights.

## Features
- **Yoga Assistant**: Real-time yoga pose detection and feedback using YOLO and gradient regressor models for pose adjustments.
- **Disease Prediction**: Predicts diseases based on symptoms using machine learning models.
- **Mental Health Analysis**: Analyzes emotional patterns and provides mental health insights using Hume AI.
- **Appointment Management**: Schedule and manage appointments.
- **Medical Records**: Secure storage and access to medical records.
- **Health Tracking**: Monitor vital signs and medications.
- **Hospital Locator**: Find nearest healthcare facilities.
- **Health Recommendations**: Personalized health recommendations based on user data.

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/ShubhxYT/Vaidya-Health-Care-with-Yoga-Assistant
    cd Vaidya-Health-Care-with-Yoga-Assistant
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```
Additionally, confirm that your nvidia card's cuda is installed and configured (AMD is not supported) and that your microphone is set up for Lumia AI.

3. Download the necessary models and place them in the `Models` directory:
    - `Gradient_Regressor_model.pkl`
    - `yolov8m.pt`

4. Ensure the following files are in the root directory for training models:
    - [Yoga.Posture.yolov8](https://universe.roboflow.com/yogaposture/yoga-posture-2/dataset/1)
    - [yoga-posture-dataset](https://www.kaggle.com/datasets/tr1gg3rtrash/yoga-posture-dataset/data)
    - [pose_landmarker_heavy/lite](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)

## Usage
Run the main application:
```sh
streamlit run app.py
```
