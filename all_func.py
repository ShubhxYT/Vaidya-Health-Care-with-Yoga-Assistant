import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import cv2
import numpy as np
import pprint as pp


# base_options = python.BaseOptions(model_asset_path='Models/pose_landmarker_heavy.task')
base_options = python.BaseOptions(model_asset_path='Models/pose_landmarker_lite.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark, NormalizedLandmarkList

def format_landmarks(landmark_data):
    # Create a list to store NormalizedLandmark objects
    landmark_data = landmark_data[0]
    landmark_list = NormalizedLandmarkList()
    # pp.pprint(landmark_data)

    for data in landmark_data:
        
        landmark = NormalizedLandmark(
            x=data.x,
            y=data.y,
            z=data.z,
            visibility=data.visibility
        )
        landmark_list.landmark.append(landmark)

    return landmark_list

def process_and_extract_keypoints(image_path):
    try:
        image = mp.Image.create_from_file(image_path)
        try:
            results = detector.detect(image)
            keypoints = extract_keypoints(results)
            return keypoints
        except RuntimeError as e:
            print(f"Error processing image {image_path}: {e}")
            return None
        
    except RuntimeError as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Function to extract keypoints
def extract_keypoints(results):
    if results.pose_landmarks:
        return [
            landmark_pb2.NormalizedLandmark(
                x=landmark.x, y=landmark.y, z=landmark.z
            )
            for landmark in results.pose_landmarks[0]
        ]
    return []

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle)
    
    return angle_degrees

# Function to calculate angles for key body parts
def calculate_angles(keypoints):
    angles = []
    if keypoints:
        left_elbow_angle = calculate_angle(keypoints[11], keypoints[13], keypoints[15])
        right_elbow_angle = calculate_angle(keypoints[12], keypoints[14], keypoints[16])
        left_shoulder_angle = calculate_angle(keypoints[23], keypoints[11], keypoints[13])
        right_shoulder_angle = calculate_angle(keypoints[24], keypoints[12], keypoints[14])
        left_hip_angle = calculate_angle(keypoints[25], keypoints[23], keypoints[11])
        right_hip_angle = calculate_angle(keypoints[26], keypoints[24], keypoints[12])
        left_knee_angle = calculate_angle(keypoints[23], keypoints[25], keypoints[27])
        right_knee_angle = calculate_angle(keypoints[24], keypoints[26], keypoints[28])
        angles = [
            left_elbow_angle, right_elbow_angle, left_shoulder_angle, right_shoulder_angle,
            left_hip_angle, right_hip_angle, left_knee_angle, right_knee_angle
        ]
    return angles

# Function to calculate the adjustment needed for each angle
def calculate_adjustments(correct_angles, user_angles):
    try:
        correct_angles = np.array([angle for angle in correct_angles], dtype=float)
        user_angles = np.array([angle for angle in user_angles], dtype=float)
        return correct_angles - user_angles
    except Exception as e:
        print(f"Error calculating adjustments: {e}")
        return []

def calculate_percentage(adjustments, lower_threshold=10.0, upper_threshold=50.0):
    angular_distances = np.abs(adjustments)
    
    # Initialize correctness_percentages to 0
    correctness_percentages = np.zeros_like(angular_distances)
    
    # Case when angular_distances are less than or equal to lower_threshold
    below_lower_threshold = angular_distances <= lower_threshold
    correctness_percentages[below_lower_threshold] = 100
    
    # Case when angular_distances are between lower_threshold and upper_threshold
    between_thresholds = (angular_distances > lower_threshold) & (angular_distances < upper_threshold)
    correctness_percentages[between_thresholds] = 100 * (1 - (angular_distances[between_thresholds] - lower_threshold) / (upper_threshold - lower_threshold))
    
    # No need to explicitly set the values for angular_distances >= upper_threshold as they are already 0
    
    # Calculate overall correctness percentage
    overall_correctness_percentage = np.mean(correctness_percentages)
    
    return overall_correctness_percentage

# Mapping of angles to body parts
body_parts = [
    "Left Elbow", "Right Elbow", "Left Shoulder", "Right Shoulder", 
    "Left Hip", "Right Hip", "Left Knee", "Right Knee"
]

# Function to provide directional feedback based on model predictions
def provide_specific_feedback(correct_pose_angles, user_pose_angles, model, threshold=5.0):
    try:
        user_pose_angles = np.array([user_pose_angles])
        correct_pose_angles = np.array([correct_pose_angles])
        
        # pp.pprint(user_pose_angles)
        # pp.pprint(correct_pose_angles)
        
        # print(calculate_adjustments(user_pose_angles,correct_pose_angles))
        
        adjustments = model.predict(calculate_adjustments(user_pose_angles,correct_pose_angles))
        # adjustments = calculate_adjustments(user_pose_angles,correct_pose_angles)

        # print(adjustments)
        
        score=calculate_percentage(adjustments)
        feedback = []
        feedback_cv = []
        for idx, (correct_angle, user_angle, adjustment) in enumerate(zip(correct_pose_angles[0], user_pose_angles[0], adjustments[0])):
            body_part = body_parts[idx]
            if abs(adjustment) < threshold:
                feedback.append(f"{body_part}: Good alignment!")
                
            else:
                direction = "decrease" if adjustment > 0 else "increase"
                detailed_direction = ""
                if body_part in ["Left Elbow", "Right Elbow"]:
                    if direction == "increase":
                        detailed_direction = "Lift your arm higher."
                    else:
                        detailed_direction = "Lower your arm."
                elif body_part in ["Left Shoulder", "Right Shoulder"]:
                    if direction == "increase":
                        detailed_direction = "Move your shoulder up."
                    else:
                        detailed_direction = "Move your shoulder down."
                elif body_part in ["Left Hip", "Right Hip"]:
                    if direction == "increase":
                        detailed_direction = "Lift your hip higher."
                    else:
                        detailed_direction = "Lower your hip."
                elif body_part in ["Left Knee", "Right Knee"]:
                    if direction == "increase":
                        detailed_direction = "Lift your knee higher."
                    else:
                        detailed_direction = "Lower your knee."
                
                feedback.append(f"{body_part}: Your angle is {user_angle:.2f}. Adjust by {abs(adjustment):.2f} degrees to {direction}. {detailed_direction}")
                feedback_cv.append(body_part)
                
        return feedback,score,feedback_cv
    except Exception as e:
        print(f"Error providing feedback: {e}")
        return [], 0.0

# Function to calculate the correctness percentage of a user's pose
def calculate_correctness_percentage(correct_angles, user_angles, lower_threshold=10.0, upper_threshold=50.0):
    correct_angles = np.array(correct_angles)
    user_angles = np.array(user_angles)
    
    # Calculate angular distances between correct and user angles
    angular_distances = np.abs(calculate_adjustments(correct_angles , user_angles))
    # Initialize correctness_percentages to 0
    correctness_percentages = np.zeros_like(angular_distances)
    
    # Case when angular_distances are less than or equal to lower_threshold
    below_lower_threshold = angular_distances <= lower_threshold
    correctness_percentages[below_lower_threshold] = 100
    
    # Case when angular_distances are between lower_threshold and upper_threshold
    between_thresholds = (angular_distances > lower_threshold) & (angular_distances < upper_threshold)
    correctness_percentages[between_thresholds] = 100 * (1 - (angular_distances[between_thresholds] - lower_threshold) / (upper_threshold - lower_threshold))
    
    # No need to explicitly set the values for angular_distances >= upper_threshold as they are already 0
    
    # Calculate overall correctness percentage
    overall_correctness_percentage = np.mean(correctness_percentages)
    
    return overall_correctness_percentage

def correct_keypoints_finder(currentClass):
    if currentClass == 'Adho-Mukha-Svanasana':
        correct_keypoints = process_and_extract_keypoints('yoga-posture-dataset/Adho Mukha Svanasana/File1.png')
    elif currentClass == 'Adho-Mukha-Vrksasana':
        correct_keypoints = process_and_extract_keypoints('yoga-posture-dataset/Adho Mukha Vrksasana/File10.png')
    elif currentClass == 'Anjaneyasana':
        correct_keypoints = process_and_extract_keypoints('yoga-posture-dataset/Anjaneyasana/File13.png')
    elif currentClass == 'Ardha-Chandrasana':
        correct_keypoints = process_and_extract_keypoints("yoga-posture-dataset/Ardha Chandrasana/File5.png")
    elif currentClass == 'Ardha-Pincha-Mayurasana':
        correct_keypoints = process_and_extract_keypoints("yoga-posture-dataset/Ardha Pincha Mayurasana/File21.png")
    elif currentClass == 'Baddha-Konasana':
        correct_keypoints = process_and_extract_keypoints("yoga-posture-dataset/Baddha Konasana/File4.png")
    elif currentClass == 'Bakasana':
        correct_keypoints = process_and_extract_keypoints("yoga-posture-dataset/Bakasana/File6.png")
    elif currentClass == 'Balasana':
        correct_keypoints = process_and_extract_keypoints("yoga-posture-dataset/Balasana/File15.png")
    elif currentClass == 'Bitilasana':
        correct_keypoints = process_and_extract_keypoints("yoga-posture-dataset/Bitilasana/File14.png")
    elif currentClass == 'Camatkarasana':
        correct_keypoints = process_and_extract_keypoints("yoga-posture-dataset/Camatkarasana/File5.png")
    elif currentClass == 'Dhanurasana':
        correct_keypoints = process_and_extract_keypoints("yoga-posture-dataset/Dhanurasana/File8.png")
    elif currentClass == 'Eka-Pada-Rajakapotasana':
        correct_keypoints = process_and_extract_keypoints("yoga-posture-dataset/Eka Pada Rajakapotasana/File7.png")
    elif currentClass == 'Garudasana':
        correct_keypoints = process_and_extract_keypoints("yoga-posture-dataset/Garudasana/File52.png")
    elif currentClass == 'Halasana':
        correct_keypoints = process_and_extract_keypoints("yoga-posture-dataset/Halasana/File3.png")
    elif currentClass == 'Hanumanasana':
        correct_keypoints = process_and_extract_keypoints("yoga-posture-dataset/Hanumanasana/File4.png")
    elif currentClass == 'Malasana':
        correct_keypoints = process_and_extract_keypoints("yoga-posture-dataset/Malasana/File11.png")
    elif currentClass == 'Marjaryasana':
        correct_keypoints = process_and_extract_keypoints("yoga-posture-dataset/Marjaryasana/File11.png")
    elif currentClass == 'Padmasana':
        correct_keypoints = process_and_extract_keypoints("yoga-posture-dataset/Padmasana/File5.png")
    elif currentClass == 'Setu-Bandha-Sarvangasana':
        correct_keypoints = process_and_extract_keypoints("yoga-posture-dataset/Setu Bandha Sarvangasana/File4.png")
    elif currentClass == 'Ustrasana':
        correct_keypoints = process_and_extract_keypoints("yoga-posture-dataset/Ustrasana/File7.png")
        
    return correct_keypoints