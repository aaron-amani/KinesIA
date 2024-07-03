import sys
import cv2
import dlib
import numpy as np
import joblib
from imutils import face_utils
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer,Qt


def write_command(command):
    """ Write the command to a text file. """
    with open("blender_commands.txt", "w") as file:
        file.write(command)

class FaceLandmarkDetector(QWidget):
    def __init__(self, model_path, predictor_path):
        super().__init__()
        self.setWindowTitle("Emotion Detector")
        self.setGeometry(100, 100, 640, 480)

        # Layout setup
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.label = QLabel(self)
        self.label.setFixedSize(840, 680)
        layout.addWidget(self.label)

        # QLabel pour afficher l'émotion détectée
        self.emotion_label = QLabel("Detected Emotion: None", self)
        self.emotion_label.setStyleSheet("font-size: 18px; color: red; background-color: white;")
        layout.addWidget(self.emotion_label, 0, alignment=Qt.AlignBottom)  # Alignement au bas

        # Initialize face detection components
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.model = joblib.load(model_path)
        self.model_points = np.array([
            (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), 
            (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0), 
            (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
        ], dtype=np.float32)
        self.camera_matrix = np.array([[650, 0, 320], [0, 650, 240], [0, 0, 1]], dtype="double")
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame")
            pass
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        
        faces = self.detector(gray, 0)
        for rect in faces:

            try :
                shape2 = self.predictor(gray, rect)
                shape_np2 = face_utils.shape_to_np(shape2)

                best_points2 = []
                for i in range(len(shape_np2)):
                    if i > 17 and (i < 27 or i > 35):
                        best_points2.append(shape_np2[i])

                # Extract the face ROI and resize it
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                face_roi = gray[y:y + h, x:x + w]
        
                face_roi_resized = cv2.resize(face_roi, (48, 48))
                # Get landmarks from the resized face ROI
                face_rect = dlib.rectangle(0, 0, 48, 48)  # Redefine face rect for the resized image
                shape = self.predictor(face_roi_resized, face_rect)
                shape_np = face_utils.shape_to_np(shape)

                best_points = []
                for i in range(len(shape_np)):
                    if i > 17 and (i < 27 or i > 35):
                        best_points.append(shape_np[i][0])
                        best_points.append(shape_np[i][1])

                
                emotion = self.predict_emotion(best_points)

                for (x, y) in best_points2:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                cv2.putText(frame, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                write_command(emotion)

                self.emotion_label.setText(f"Detected Emotion: {emotion.capitalize()}")
                
            except:
                pass


        self.display_image(frame)

    def display_image(self, frame):
        """ Convert image to RGB and display it. """
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.label.width(), self.label.height(), aspectRatioMode = 1)
        self.label.setPixmap(QPixmap.fromImage(p))

    def predict_emotion(self, landmarks):
        all_features = np.array(landmarks)
        
        all_features = all_features.reshape(1, -1)
        return self.model.predict(all_features)[0]
    
    def closeEvent(self, event):
        self.cap.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceLandmarkDetector("RF_emotion_classifier_24_042.pkl", "shape_predictor_68_face_landmarks.dat")
    window.show()
    sys.exit(app.exec_())
