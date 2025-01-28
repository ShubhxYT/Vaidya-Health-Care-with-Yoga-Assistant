import os

import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split



from all_func import *


dataset_path = 'yoga-posture-dataset'
angles_set=[]
angles_label=[]
file_paths=[]
labels=[]
for folder in  os.listdir(dataset_path):
    cls_folder= os.path.join(dataset_path, folder)
    if os.path.isdir(cls_folder):
        for img_name in os.listdir(cls_folder):
            imagepath=os.path.join(cls_folder, img_name)
            # print('\n\n',end="\r")
            print(f"{imagepath}                             ",end="\r")
            # print('\n\n',end="\r")
            keypoints = process_and_extract_keypoints(imagepath)
            if keypoints:
                angles = calculate_angles(keypoints)
                angles_set.append(angles)
                angles_label.append(folder)
    #             file_paths.append(os.path.join(cls_folder, img_name))
    #             labels.append(folder)
    
# Prepare the dataset
label=[]
correct_angle=[]
ds_angles = []
ds_adjustments = []
ds_scores=[]
for folder in  os.listdir(dataset_path):
    i=0
    if os.path.isdir(cls_folder):
        for label,user_angles in zip(angles_label,angles_set):
            if i==0:
                correct_angle = user_angles
                i+=1
            if label == folder:
                adjustments = calculate_adjustments(correct_angle, user_angles)
                ds_angles.append(adjustments)
                ds_adjustments.append(np.zeros((8,)))
                ds_scores.append(100)
            else:
                adjustments = calculate_adjustments(correct_angle, user_angles)
                ds_angles.append(adjustments)
                ds_adjustments.append(adjustments)
                score = calculate_percentage(adjustments)
                ds_scores.append(score)
                
    

ds_angles = np.array(ds_angles)
ds_adjustments = np.array(ds_adjustments)

# Save the arrays to .npy files
np.save('ds_angles.npy', ds_angles)
np.save('ds_adjustments.npy', ds_adjustments)
