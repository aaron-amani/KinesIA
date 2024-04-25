import os
import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imutils import face_utils
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit


# Path to the facial landmark predictor model
predictor_path = "models/shape_predictor_68_face_landmarks.dat"

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Define model points in 3D space
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float32)

def extract_landmarks_and_pose(image_path):
    """ Extract facial landmarks and compute face pose from a given image """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load {image_path}")
        return [], None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1) #What if the image is already grey
    best_points = []

    for rect in rects:
        shape = predictor(gray, rect)
        shape_np = face_utils.shape_to_np(shape)
        if image.shape[0] != 48 or image.shape[1] != 48:
            try:
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                face_roi = gray[y:y + h, x:x + w]
                face_roi_resized = cv2.resize(face_roi, (48, 48))
                face_rect = dlib.rectangle(0, 0, 48, 48)  # Redefine face rect for the resized image
                shape = predictor(face_roi_resized, face_rect)
                shape_np = face_utils.shape_to_np(shape)
            except :
                print("Impossible to resize image !")
        for i in range(len(shape_np)):
            if i > 17 and (i < 27 or i > 35):
                best_points.append(shape_np[i][0])
                best_points.append(shape_np[i][1])

    return best_points

def prepare_datas(file_path_list) -> pd.DataFrame:
    """ Organize data into a DataFrame with labels and poses """
    datas = []
    labels = []

    for i in tqdm(range(len(file_path_list))):
        landmarks = extract_landmarks_and_pose(str(file_path_list[i]))
        if landmarks != []:
            datas.append(landmarks)
            labels.append(str(file_path_list[i]).split('/')[2])

    feature_data = pd.DataFrame(datas)
    #pose_data = pd.DataFrame(all_poses, columns=['rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z'])
    feature_data['emotion'] = labels
    return feature_data

def data_prep_split(datas: pd.DataFrame) -> tuple:
    """ Train models """
    y = datas["emotion"]
    X = datas.drop(columns="emotion")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    for (train_index, test_index) in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    return X_train, X_test, y_train, y_test