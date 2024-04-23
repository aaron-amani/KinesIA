import os
import cv2
import dlib
import numpy as np
import pandas as pd
from imutils import face_utils
import pathlib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

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
    rects = detector(gray, 1)
    all_landmarks = []
    poses = []

    for rect in rects:
        shape = predictor(gray, rect)
        shape_np = face_utils.shape_to_np(shape)
        
        # Define image points which are analogous to model points
        image_points = np.array([
            shape_np[30],  # Nose tip
            shape_np[8],   # Chin
            shape_np[36],  # Left eye left corner
            shape_np[45],  # Right eye right corner
            shape_np[48],  # Left Mouth corner
            shape_np[54]   # Right mouth corner
        ], dtype=np.float32)

        # Camera matrix
        size = image.shape
        focal_length = size[1]
        center = (size[1]//2, size[0]//2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        # Compute the pose of the face using solvePnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE)
        if success:
            all_landmarks.append(shape_np.flatten())
            poses.append(np.concatenate((rotation_vector.flatten(), translation_vector.flatten())))
        return all_landmarks, poses

def prepare_data_datas_00():
    """ Organize data into a DataFrame with labels and poses """
    data = []
    labels = []
    all_poses = []

    file_path_list = list(pathlib.Path("datas#00").glob('*/*.png'))
    for i in tqdm(range(len(file_path_list))):
        landmarks, poses = extract_landmarks_and_pose(str(file_path_list[i]))
        for landmark_set, pose_set in zip(landmarks, poses):
            data.append(landmark_set)
            labels.append(str(file_path_list[i].stem).split('_')[3])
            all_poses.append(pose_set)

    feature_data = pd.DataFrame(data)
    pose_data = pd.DataFrame(all_poses, columns=['rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z'])
    feature_data = pd.concat([feature_data, pose_data], axis=1)
    feature_data['emotion'] = labels
    return feature_data



# Directory containing the dataset organized by emotion
#data_directory = "datas#00/"
datasetcsv_path = "features_with_pose.csv"
features = prepare_data_datas_00()


# Save features and labels to a CSV file for later use
if not pathlib.Path(datasetcsv_path).exists:
    features.to_csv(datasetcsv_path, index=False)
    print("Data extraction and pose estimation complete. Features saved to 'features_with_pose.csv'.")