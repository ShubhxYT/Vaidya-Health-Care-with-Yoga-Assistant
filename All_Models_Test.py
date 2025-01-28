import numpy as np
import joblib
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from all_func import *

ds_angles = np.load('data/ds_angles.npy')
ds_adjustments = np.load('data/ds_adjustments.npy')

pp.pprint(ds_angles)
print("==========================")
pp.pprint(ds_adjustments)

train_angles, test_angles, train_adjustments, test_adjustments = train_test_split(ds_angles, ds_adjustments, test_size=0.2, random_state=42)

# Load and process both images
correct_keypoints = process_and_extract_keypoints('yoga-posture-dataset/Adho Mukha Svanasana/File1.png')
user_keypoints =  process_and_extract_keypoints('yoga-posture-dataset/Anjaneyasana/File11.png')
# user_keypoints = process_and_extract_keypoints('yoga-posture-dataset/Adho Mukha Svanasana/File21.png')

new_correct_pose_angles = calculate_angles(correct_keypoints)
new_user_pose_angles = calculate_angles(user_keypoints)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

regressor = joblib.load('Models/RandomForest_Regressor_model.pkl')
multi_target_regressor = joblib.load("Models/Gradient_Regressor_model.pkl")
model = joblib.load("Models/neural_network_model.pkl")
# ================== Random Forest Regressor ====================
t = time.time()
print("\n\n\n Random Forest Regressor \n")

# pred_adjustments = regressor.predict(test_angles)# Evaluate the model
# mse = mean_squared_error(test_adjustments, pred_adjustments)
# print(f'Mean Squared Error: {mse:.2f}')

feedback,score = provide_specific_feedback(new_correct_pose_angles, new_user_pose_angles, regressor)
for message in feedback:
    print(message)
    
print(f"Overall Pose Correctness Predict By Model: {score:.2f}%")

print(f"time took : {(time.time()) -t}")


# ================== Gradient Regressor ===========================
t = time.time()
print("\n\n\n Gradient Regressor \n")
# Predict on the test set
# pred_adjustments = multi_target_regressor.predict(test_angles)

# # Evaluate the model
# mse = mean_squared_error(test_adjustments, pred_adjustments)
# print(f'MultiOutput Gradient Boosting Regressor Mean Squared Error: {mse:.2f}')

# test calculate angles
# Load and process both images

feedback,score = provide_specific_feedback(new_correct_pose_angles, new_user_pose_angles, multi_target_regressor)
for message in feedback:
    print(message)
print(f"Overall Pose Correctness Predict By Model: {score:.2f}%")
print(f"time took : {((time.time()) -t)}")

# ================== Neural Network ===================================
t = time.time()
print("\n\n\n Neural Network \n")

# pred_adjustments_nn = model.predict(test_angles)

# mse_nn = mean_squared_error(test_adjustments, pred_adjustments_nn)
# print(f'Neural Network Mean Squared Error: {mse_nn:.2f}')

feedback,score = provide_specific_feedback(new_correct_pose_angles, new_user_pose_angles, model)
for message in feedback:
    print(message)

print(f"Overall Pose Correctness: {score:.2f}%")
print(f"time took : {(time.time()) -t}")
print("\n\n\n")
overall_correctness_percentage = calculate_correctness_percentage(new_correct_pose_angles, new_user_pose_angles)
print(f"Overall Correctness Percentage: {overall_correctness_percentage:.2f}%")


