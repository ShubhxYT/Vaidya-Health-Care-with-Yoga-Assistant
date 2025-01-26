import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from all_func import *

# Load the arrays from .npy files
ds_angles = np.load('ds_angles.npy')
ds_adjustments = np.load('ds_adjustments.npy')

train_angles, test_angles, train_adjustments, test_adjustments = train_test_split(ds_angles, ds_adjustments, test_size=0.2, random_state=42)


# Train a Random Forest Regressor
regressor = RandomForestRegressor(n_estimators=500, random_state=42)
regressor.fit(train_angles, train_adjustments)

import joblib
joblib.dump(regressor, 'Gradient_Regressor_model.pkl')

# Predict on the test set
pred_adjustments = regressor.predict(test_angles)# Evaluate the model
mse = mean_squared_error(test_adjustments, pred_adjustments)
print(f'Mean Squared Error: {mse:.2f}')

# Load and process both images
correct_keypoints = process_and_extract_keypoints('yoga-posture-dataset/Adho Mukha Svanasana/File1.png')
# user_keypoints =  process_and_extract_keypoints('yoga-posture-dataset/Anjaneyasana/File11.png')
user_keypoints = process_and_extract_keypoints('yoga-posture-dataset/Adho Mukha Svanasana/File21.png')

# Calculate angles for both poses
new_correct_pose_angles = calculate_angles(correct_keypoints)
new_user_pose_angles = calculate_angles(user_keypoints)
# regressor=0
feedback,score = provide_specific_feedback(new_correct_pose_angles, new_user_pose_angles, regressor)
for message in feedback:
    print(message)
    
print(f"Overall Pose Correctness Predict By Model: {score:.2f}%")

overall_correctness_percentage = calculate_correctness_percentage(new_correct_pose_angles, new_user_pose_angles)
print(f"Overall Pose Correctness: {overall_correctness_percentage:.2f}%")