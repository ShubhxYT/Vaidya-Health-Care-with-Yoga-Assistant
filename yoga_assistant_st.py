# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import cv2
import numpy as np
import pprint as pp
import cv2
import mediapipe as mp
import time
import streamlit as st

from all_func import *
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

from ultralytics import YOLO
import cvzone
import math

name_lst = ['Adho-Mukha-Svanasana', 'Adho-Mukha-Vrksasana', 'Anjaneyasana', 'Ardha-Chandrasana', 'Ardha-Pincha-Mayurasana', 'Baddha-Konasana', 'Bakasana', 'Balasana', 'Bitilasana', 'Camatkarasana', 'Dhanurasana', 'Eka-Pada-Rajakapotasana', 'Garudasana', 'Halasana', 'Hanumanasana', 'Malasana', 'Marjaryasana', 'Padmasana', 'Setu-Bandha-Sarvangasana', 'Ustrasana']



# base_options = python.BaseOptions(model_asset_path='Models/pose_landmarker_heavy.task')
base_options = python.BaseOptions(model_asset_path='Models/pose_landmarker_lite.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)



def video_stream(x):
    """
    Handle the video stream and exercise tracking.
    """
    # Create stop button
    stop_button = st.button('Stop')
    video_placeholder = st.empty()
    
    try:
        
        # wCam, hCam = 1280, 720
        if x == "cam":
            cap = cv2.VideoCapture(0)
        elif x == "test":
            cap = cv2.VideoCapture("videoplayback.mp4") #any sample yoga video
            
        if not cap.isOpened():
            st.error("Cannot access webcam. Please check your camera connection.")
            return

        mpPose = mp.solutions.pose
        mpDraw = mp.solutions.drawing_utils
        formated_out_old = NormalizedLandmarkList()
        for i in range(33):
            landmark = NormalizedLandmark(
                        x=0,
                        y=0,
                        z=0,
                        visibility=0
                    )
            formated_out_old.landmark.append(landmark)
        ptime = 0

        model = joblib.load("Models/Gradient_Regressor_model.pkl")
        # model = joblib.load("Models/neural_network_model_2000.pkl")
        yolo_model = YOLO('Models/yolo-yoga.pt')

        feedback_cv = []

        while stop_button == False:
            try:
                success , img = cap.read()
                imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
                
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)

                results_mp = detector.detect(mp_img)
                copy_img = img.copy()
                yolo_results = yolo_model(copy_img)
                score = 0
                
                for result in yolo_results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Confidence
                        yolo_conf = math.ceil((box.conf[0] * 100)) / 100
                        yolo_cls = int(box.cls[0])
                        currentClass = name_lst[yolo_cls]
                                    
                        # Class Names
                        if yolo_conf > .40:
                            # cvzone.putTextRect(img, f'{yolo_conf}:{currentClass}', (max(0, x1), max(30-100, y1)), scale=1, thickness=1)
                            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            height, width = img.shape[:2]
                            cv2.putText(img, f'{yolo_conf}:{currentClass}', (25, height-15), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 4)
                            
                            correct_keypoints = correct_keypoints_finder(currentClass)
                            
                try :
                    formated_out = format_landmarks(results_mp.pose_landmarks)
                    # print(formated_out) 
                    formated_out_old = formated_out
                except :
                    formated_out = formated_out_old
                
                try:
                    # correct_keypoints = process_and_extract_keypoints('yoga-posture-dataset/Adho Mukha Svanasana/File1.png')
                    user_keypoints =  extract_keypoints(results_mp)
                    new_correct_pose_angles = calculate_angles(correct_keypoints)
                    new_user_pose_angles = calculate_angles(user_keypoints)
                    
                    
                    t = time.time()
                    print("\n\n\n model \n")
                    try :
                        if yolo_conf > .70:
                            feedback,score,feedback_cv = provide_specific_feedback(new_correct_pose_angles, new_user_pose_angles, model)
                            for message in feedback:
                                print(message)
                        # print(feedback_cv)
                    
                        print(f"Overall Pose Correctness Predict By Model: {score:.2f}%")
                        
                        print(f"time took : {((time.time()) -t)}")
                    except Exception as e:
                        print(f"Error processing image {e}")
                except Exception as e:
                    print(f"Error processing image {e}")
                
                
                
                mpDraw.draw_landmarks(img , formated_out,mpPose.POSE_CONNECTIONS)
                for id,lm in enumerate(formated_out.landmark):
                    h,w,c = img.shape
                    # print(id , lm)
                    cx , cy = int(lm.x * w) , int(lm.y *h)
                    # cv2.circle(img,(cx,cy),3,(255,0,0),cv2.FILLED)
                    
                    for lst in feedback_cv:
                        if score <= 70:
                            # print(f'\n\n{score}\n\n')
                            if lst == "Left Elbow" and id == 13:
                                cv2.circle(img,(cx,cy),18,(255, 0,255),cv2.FILLED)
                            elif lst == "Right Elbow" and id == 14:
                                cv2.circle(img,(cx,cy),18,(255, 0,255),cv2.FILLED)
                            elif lst == "Left Shoulder" and id == 11:
                                cv2.circle(img,(cx,cy),18,(255, 0,255),cv2.FILLED)
                            elif lst == "Right Shoulder" and id == 12:
                                cv2.circle(img,(cx,cy),18,(255, 0,255),cv2.FILLED)
                            elif lst == "Left Hip" and id == 23:
                                cv2.circle(img,(cx,cy),18,(255, 0,255),cv2.FILLED)
                            elif lst == "Right Hip" and id == 24:
                                cv2.circle(img,(cx,cy),18,(255, 0,255),cv2.FILLED)
                            elif lst == "Left Knee" and id == 25:
                                cv2.circle(img,(cx,cy),18,(255, 0,255),cv2.FILLED)
                            elif lst == "Right Knee" and id == 26:
                                cv2.circle(img,(cx,cy),18,(255, 0,255),cv2.FILLED)
                        elif score <=65:
                            feedback_cv = []
                        else:
                            if id == 13:
                                cv2.circle(img,(cx,cy),25,(0,255,0),cv2.FILLED)
                            elif id == 14:
                                cv2.circle(img,(cx,cy),25,(0,255,0),cv2.FILLED)
                            elif id == 11:
                                cv2.circle(img,(cx,cy),25,(0,255,0),cv2.FILLED)
                            elif id == 12:
                                cv2.circle(img,(cx,cy),25,(0,255,0),cv2.FILLED)
                            elif id == 23:
                                cv2.circle(img,(cx,cy),25,(0,255,0),cv2.FILLED)
                            elif id == 24:
                                cv2.circle(img,(cx,cy),25,(0,255,0),cv2.FILLED)
                            elif id == 25:
                                cv2.circle(img,(cx,cy),25,(0,255,0),cv2.FILLED)
                            elif id == 26:
                                cv2.circle(img,(cx,cy),25,(0,255,0),cv2.FILLED)
                    
                # overall_correctness_percentage = calculate_correctness_percentage(new_correct_pose_angles, new_user_pose_angles)
                # print(f"Overall Correctness Percentage: {overall_correctness_percentage:.2f}%")
                
                
                ctime = time.time()
                fps = 1 / (ctime-ptime)
                ptime = ctime
                # imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
                
                # frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # img = Image.fromarray(imgRGB)
                cv2.putText(img , str(int(fps)) , (70,50) , cv2.FONT_HERSHEY_PLAIN , 3, (255,0,0),3)
                
                video_placeholder.image(img, caption="Exercise Tracker", use_container_width=True)
                
                
                # cv2.imshow("Image",img)
                # cv2.imshow("Image",img)

                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                
            except RuntimeError as e:
                print(f"Error processing : {e}")  
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        
    finally:
        # Clean up
        if 'cap' in locals():
            cap.release()  
            
            
def app():
    """
    Main Streamlit application.
    """
    st.title("Coach Vaidya - Yoga Pose Tracker")
    st.write("This application tracks your yoga poses in real-time.")
    st.write("Make sure your full body is visible in the camera frame.")
    
    # Instructions
    
    # Start button
    if st.button('Start', key='start_video'):
        video_stream("cam")
    if st.button('Test Assistant'):
        video_stream("test")

if __name__ == "__main__":
    app()
