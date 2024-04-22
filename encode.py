import cv2 #librairie OpenCV
import dlib # détection de visages et la reconnaissance faciale.
import numpy as np # calcul numérique
from imutils import face_utils
import pandas as pd
import pathlib


#Chargement des modele de predictions 
pose_predictor_68_point = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")
pose_predictor_5_point = dlib.shape_predictor("./models/shape_predictor_5_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("./models/dlib_face_recognition_resnet_model_v1.dat")
face_detector = dlib.get_frontal_face_detector()


#La fonction transform transforme les coordonnées des visages détectés pour qu'elles restent à l'intérieur de l'image.
def transform(image, face_locations):
    coord_faces = []
    for face in face_locations:
        rect = face.top(), face.right(), face.bottom(), face.left()
        coord_face = max(rect[0], 0), min(rect[1], image.shape[1]), min(rect[2], image.shape[0]), max(rect[3], 0)
        coord_faces.append(coord_face)
    return coord_faces

#La fonction encode_face détecte les visages dans l'image, calcule les descripteurs de chaque visage et récupère les points caractéristiques.
def encode_face(image):
    face_locations = face_detector(image, 1)
    face_encodings_list = []
    landmarks_list = []
    for face_location in face_locations:
        # DETECT FACES
        shape = pose_predictor_68_point(image, face_location)
        face_encodings_list.append(np.array(face_encoder.compute_face_descriptor(image, shape, num_jitters=1)))
        # GET LANDMARKS
        shape = face_utils.shape_to_np(shape)
        landmarks_list.append(shape)
    face_locations = transform(image, face_locations)
    return face_encodings_list, face_locations, landmarks_list




vec_target_dict = {"vector": [], "target": []}
for path_img in list(pathlib.Path("datas").glob('*/*.png')):
    image = cv2.imread(str(path_img))
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_encodings_list, face_locations, landmarks_list = encode_face(rgb_image)
    vec_target_dict["vector"].append(face_encodings_list)
    vec_target_dict["target"].append(str(path_img).split('_')[2])
#print(face_encodings_list)
print(vec_target_dict["target"][0], vec_target_dict["vector"][0])
