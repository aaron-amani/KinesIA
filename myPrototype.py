import bpy
from imutils import face_utils
import sys
#sys.path.insert(0,'/Users/aaron_amani/Library/Python/3.7/lib/python/site-packages')
import dlib
import cv2
import time
import numpy
#from bpy.props import FloatProperty
import joblib
    
class OpenCVAnimOperator(bpy.types.Operator):
    """Operator which runs its self from a timer"""
    bl_idname = "wm.opencv_operator"
    bl_label = "OpenCV Animation Operator"
    
    model = joblib.load("/Users/aaron_amani/Workspace/Projet Recherche/KinesIA/RF_emotion_classifier_24_042.pkl")
     
    # p = our pre-treined model directory
    #p = "D:\\Users\\jason\\shape_predictor_68_face_landmarks.dat" # Windows
    p = "/Users/aaron_amani/Downloads/blender/shape_predictor_68_face_landmarks.dat" # macOS
    #p = "/home/jason/Downloads/shape_predictor_68_face_landmarks.dat" # linux
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)
    # rig_name - take it from your scene collection tree
    rig_name = "RIG-Vincent"

    _timer = None
    _cap  = None
    
    width = 600
    height = 400

    stop :bpy.props.BoolProperty()
    
    # 3D model points.    
    model_points = numpy.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                            ], dtype = numpy.float32)
    # Camera internals
    camera_matrix = numpy.array(
                            [[height, 0.0, width/2],
                            [0.0, height, height/2],
                            [0.0, 0.0, 1.0]], dtype = numpy.float32
                            )
    
    # Keeps a moving average of given length
    def smooth_value(self, name, length, value):
        if not hasattr(self, 'smooth'):
            self.smooth = {}
        if not name in self.smooth:
            self.smooth[name] = numpy.array([value])
        else:
            self.smooth[name] = numpy.insert(arr=self.smooth[name], obj=0, values=value)
            if self.smooth[name].size > length:
                self.smooth[name] = numpy.delete(self.smooth[name], self.smooth[name].size-1, 0)
        sum = 0
        for val in self.smooth[name]:
            sum += val
        return sum / self.smooth[name].size


    # Keeps min and max values, then returns the value in a ranve 0 - 1
    def get_range(self, name, value):
        if not hasattr(self, 'range'):
            self.range = {}
        if not name in self.range:
            self.range[name] = numpy.array([value, value])
        else:
            self.range[name] = numpy.array([min(value, self.range[name][0]), max(value, self.range[name][1])] )
        val_range = self.range[name][1] - self.range[name][0]
        if val_range != 0:
            return (value - self.range[name][0]) / val_range
        else:
            return 0

    def modal(self, context, event):

        if (event.type in {'RIGHTMOUSE', 'ESC'}) or self.stop == True:
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'TIMER':
            self.init_camera()
            _, image = self._cap.read()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 0)
            # bpy.context.scene.frame_set(frame_num)
         
            # For each detected face, find the landmark.
            for (i, rect) in enumerate(rects):
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
                
                self.apply_command_to_rig(emotion)
             
                for (x, y) in shape_np2:
                    cv2.circle(image, (x, y), 2, (0, 255, 255), -1)

                cv2.putText(image, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                 
                #2D image points. If you change the image, you need to change vector
                image_points = numpy.array([shape_np2[30],     # Nose tip - 31
                                            shape_np2[8],      # Chin - 9
                                            shape_np2[36],     # Left eye left corner - 37
                                            shape_np2[45],     # Right eye right corne - 46
                                            shape_np2[48],     # Left Mouth corner - 49
                                            shape_np2[54]      # Right mouth corner - 55
                                        ], dtype = numpy.float32)
             
                dist_coeffs = numpy.zeros((4,1)) # Assuming no lens distortion
             
                if hasattr(self, 'rotation_vector'):
                    (success, self.rotation_vector, self.translation_vector) = cv2.solvePnP(self.model_points, 
                        image_points, self.camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE, 
                        rvec=self.rotation_vector, tvec=self.translation_vector, 
                        useExtrinsicGuess=True)
                else:
                    (success, self.rotation_vector, self.translation_vector) = cv2.solvePnP(self.model_points, 
                        image_points, self.camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE, 
                        useExtrinsicGuess=False)
             
                if not hasattr(self, 'first_angle'):
                    self.first_angle = numpy.copy(self.rotation_vector)
             
                bones = bpy.data.objects[self.rig_name].pose.bones
                 
                bones["mouth_ctrl"].location[2] = self.smooth_value("m_h", 2, -self.get_range("mouth_height", numpy.linalg.norm(shape_np2[62] - shape_np2[66])) * 0.06 )
                bones["mouth_ctrl"].location[0] = self.smooth_value("m_w", 2, (self.get_range("mouth_width", numpy.linalg.norm(shape_np2[54] - shape_np2[48])) - 0.5) * -0.04)
                bones["brow_ctrl_L"].location[2] = self.smooth_value("b_l", 3, (self.get_range("brow_left", numpy.linalg.norm(shape_np2[19] - shape_np2[27])) -0.5) * 0.04)
                bones["brow_ctrl_R"].location[2] = self.smooth_value("b_r", 3, (self.get_range("brow_right", numpy.linalg.norm(shape_np2[24] - shape_np2[27])) -0.5) * 0.04)
                
                bones["mouth_ctrl"].keyframe_insert(data_path="location", index=-1)
                bones["brow_ctrl_L"].keyframe_insert(data_path="location", index=2)
                bones["brow_ctrl_R"].keyframe_insert(data_path="location", index=2)
                
                 
            cv2.imshow("Output", image)
            cv2.waitKey(1)

        return {'PASS_THROUGH'}
    
    def predict_emotion(self, landmarks):
        all_features = numpy.array(landmarks)
        
        all_features = all_features.reshape(1, -1)
        
        return self.model.predict(all_features)[0]
    
    
    def apply_command_to_rig(self, command):
        rig = bpy.data.objects.get("RIG-Vincent")
        if not rig:
            print("Rig not found.")
            return
        
        # Reset rig to neutral position before applying new expression
        self.reset_rig(rig)

        if command == "happy":
            # Assume happy involves a smile
            rig.pose.bones["mouth_ctrl"].location[2] -= 0.06  # mouth moves up slightly
            rig.pose.bones["mouth_ctrl"].location[0] += 0.02  # mouth widens

        elif command == "sad":
            # Assume sad involves a frown
            rig.pose.bones["mouth_ctrl"].location[2] += 0.06  # mouth moves down
            rig.pose.bones["mouth_ctrl"].location[0] -= 0.02  # mouth narrows

        elif command == "neutral":
            # Neutral is the rest position, already set by reset_rig
            pass

        elif command == "surprised":
            # Assume surprised involves raised eyebrows and an open mouth
            rig.pose.bones["brow_ctrl_L"].location[2] += 0.04
            rig.pose.bones["brow_ctrl_R"].location[2] += 0.04
            rig.pose.bones["mouth_ctrl"].location[2] -= 0.04  # open mouth

        elif command == "anger":
            # Assume anger involves eyebrows down and a scowl
            rig.pose.bones["brow_ctrl_L"].location[2] -= 0.04
            rig.pose.bones["brow_ctrl_R"].location[2] -= 0.04
            rig.pose.bones["mouth_ctrl"].location[2] += 0.03  # mouth moves down
            rig.pose.bones["mouth_ctrl"].location[0] -= 0.03  # mouth narrows
            
    def reset_rig(self, rig):
        for bone in rig.pose.bones:
            bone.location = (0, 0, 0)
            bone.rotation_quaternion = (1, 0, 0, 0)
            bone.rotation_euler = (0, 0, 0)
            bone.scale = (1, 1, 1)
    
    def init_camera(self):
        if self._cap == None:
            self._cap = cv2.VideoCapture(1)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            time.sleep(0.5)
        
        self.reset_rig(bpy.data.objects[self.rig_name])
    
    def stop_playback(self, scene):
        print(format(scene.frame_current) + " / " + format(scene.frame_end))
        if scene.frame_current == scene.frame_end:
            bpy.ops.screen.animation_cancel(restore_frame=False)
        
    def execute(self, context):
        bpy.app.handlers.frame_change_pre.append(self.stop_playback)

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.02, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        cv2.destroyAllWindows()
        self._cap.release()
        self._cap = None
        self.reset_rig(bpy.data.objects[self.rig_name])

def register():
    bpy.utils.register_class(OpenCVAnimOperator)

def unregister():
    bpy.utils.unregister_class(OpenCVAnimOperator)

if __name__ == "__main__":
    register()

    # test call
    #bpy.ops.wm.opencv_operator()



