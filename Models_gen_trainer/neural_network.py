from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


import tensorflow as tf
import joblib

from all_func import *

ds_angles = np.load('ds_angles.npy')
ds_adjustments = np.load('ds_adjustments.npy')

train_angles, test_angles, train_adjustments, test_adjustments = train_test_split(ds_angles, ds_adjustments, test_size=0.2, random_state=42)


model = Sequential([
    Dense(128, activation='elu', input_shape=(train_angles.shape[1],)),
    Dropout(0.2),  # 20% dropout
    Dense(128, activation='elu'),
    Dropout(0.2),  # 20% dropout
    Dense(64, activation='elu'),
    Dropout(0.1),  # 10% dropout
    Dense(train_adjustments.shape[1])  # Multi-output regression
])


# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model with early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
# model.fit(train_angles, train_adjustments, epochs=100, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping])
model.fit(train_angles, train_adjustments, epochs=2000, batch_size=32, validation_split=0.2, verbose=1)

joblib.dump(model, 'neural_network_model.pkl')
# Predict on the test set
pred_adjustments_nn = model.predict(test_angles)

# Denormalize the predictions

# Evaluate the model
mse_nn = mean_squared_error(test_adjustments, pred_adjustments_nn)
print(f'Neural Network Mean Squared Error: {mse_nn:.2f}')

# test calculate angles
# Load and process both images
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

correct_keypoints = process_and_extract_keypoints('yoga-posture-dataset/Adho Mukha Svanasana/File1.png')
# user_keypoints =  process_and_extract_keypoints('yoga-posture-dataset/Anjaneyasana/File11.png')
user_keypoints = process_and_extract_keypoints('yoga-posture-dataset/Adho Mukha Svanasana/File21.png')

# Calculate angles for both poses
new_correct_pose_angles = calculate_angles(correct_keypoints)
new_user_pose_angles = calculate_angles(user_keypoints)
# print(new_correct_pose_angles)
# print(new_user_pose_angles)

feedback,score = provide_specific_feedback(new_correct_pose_angles, new_user_pose_angles, model)
for message in feedback:
    print(message)

print(f"Overall Pose Correctness: {score:.2f}%")

overall_correctness_percentage = calculate_correctness_percentage(new_correct_pose_angles, new_user_pose_angles)
print(f"Overall Correctness Percentage: {overall_correctness_percentage:.2f}%")
