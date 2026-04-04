import math

import cv2
import mediapipe as mp
import numpy as np

# Landmark indices for Left and Right Eyes from MediaPipe Face Mesh
# Contours form the outer shape of the eye
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def compute_ear(eye_landmarks, frame_width, frame_height):
    # Convert normalized landmarks to pixel coordinates
    points = []
    for lm in eye_landmarks:
        x_px = min(math.floor(lm.x * frame_width), frame_width - 1)
        y_px = min(math.floor(lm.y * frame_height), frame_height - 1)
        points.append((x_px, y_px))

    # Compute Euclidean distances between the vertical eye landmarks
    p2_p6 = euclidean_distance(points[1], points[5])
    p3_p5 = euclidean_distance(points[2], points[4])
    # Compute Euclidean distance between horizontal eye landmarks
    p1_p4 = euclidean_distance(points[0], points[3])

    # Compute Eye Aspect Ratio
    ear = (p2_p6 + p3_p5) / (2.0 * p1_p4)
    return ear

class LivenessDetector:
    def __init__(self, ear_threshold=0.20, frames_to_blink=2):
        self.ear_threshold = ear_threshold
        self.frames_to_blink = frames_to_blink
        
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        import os

        # Check if model downloaded, otherwise download it
        model_path = 'face_landmarker.task'
        if not os.path.exists(model_path):
            import urllib.request
            url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
            print("Downloading face_landmarker model...")
            urllib.request.urlretrieve(url, model_path)
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                               output_face_blendshapes=False,
                                               output_facial_transformation_matrixes=False,
                                               num_faces=1,
                                               min_face_detection_confidence=0.5,
                                               min_face_presence_confidence=0.5,
                                               min_tracking_confidence=0.5)
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
        
        self.blink_counters = {} # Maps user/face ID to consecutive frames eye is closed
        self.liveness_state = {} # True if the user successfully blinked

    def reset(self):
        self.blink_counters = {}
        self.liveness_state = {}

    def process_frame(self, frame, user_name="current_user"):
        import math
        # Convert BGR to RGB since Mediapipe uses RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = self.face_landmarker.detect(mp_image)
        
        if user_name not in self.blink_counters:
            self.blink_counters[user_name] = 0
            self.liveness_state[user_name] = False

        # If we already passed liveness, return True immediately to save computing
        if self.liveness_state[user_name]:
            return True, None

        if results.face_landmarks:
            for face_landmarks in results.face_landmarks:
                h, w, _ = frame.shape
                
                # Get the eye landmarks
                left_eye_points = [face_landmarks[p] for p in LEFT_EYE_INDICES]
                right_eye_points = [face_landmarks[p] for p in RIGHT_EYE_INDICES]
                
                left_ear = compute_ear(left_eye_points, w, h)
                right_ear = compute_ear(right_eye_points, w, h)
                
                avg_ear = (left_ear + right_ear) / 2.0
                
                # Check for blink
                if avg_ear < self.ear_threshold:
                    self.blink_counters[user_name] += 1
                else:
                    if self.blink_counters[user_name] >= self.frames_to_blink:
                        # Eye was closed for enough frames and now is open -> BlINK
                        self.liveness_state[user_name] = True
                    self.blink_counters[user_name] = 0
                
                return self.liveness_state[user_name], avg_ear
                
        return False, None
