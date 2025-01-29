# Models Generation Trainer

This folder contains scripts and tools for training various machine learning models used in the project. The models are trained for different purposes such as pose estimation, disease prediction, and more.

## Folder Structure
- `data_generator_mediapipe.py`: Script to obtain training data for those models that are being trained; these files are already there in the directory in .npy format in data folder.
- `GradientBoostingRegressor.py`: Script to train a Gradient Boosting Regressor model.
- `neural_network.py`: Script to train a Neural Network model.
- `RandomForestRegressor.py`: Script to train a Random Forest Regressor model.
- `yolo-training.py`: Script to train a YOLO model for object detection.

## Requirements

Ensure you have the following dependencies installed:

- NumPy
- scikit-learn
- TensorFlow
- PyTorch
- OpenCV
- joblib
- ultralytics (for YOLO)

You can install the required packages using:

```sh
pip install -r requirements.txt
```
Make sure Nvidia Cuda is configured for model training as well, and select a Yolo model in accordance with your vram assumption.

