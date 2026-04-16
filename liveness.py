import math
import cv2
import mediapipe as mp
import numpy as np
from moire_detector import MoireDetector

LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def compute_ear(eye_landmarks, frame_width, frame_height):
    points = []
    for lm in eye_landmarks:
        x_px = min(math.floor(lm.x * frame_width), frame_width - 1)
        y_px = min(math.floor(lm.y * frame_height), frame_height - 1)
        points.append((x_px, y_px))

    p2_p6 = euclidean_distance(points[1], points[5])
    p3_p5 = euclidean_distance(points[2], points[4])
    p1_p4 = euclidean_distance(points[0], points[3])

    if p1_p4 == 0: return 0
    return (p2_p6 + p3_p5) / (2.0 * p1_p4)

class LivenessDetector:
    def __init__(self, ear_threshold=0.20, frames_to_blink=2):
        self.ear_threshold = ear_threshold
        self.frames_to_blink = frames_to_blink
        self.moire_detector = MoireDetector(threshold=190)
        
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
        
        self.challenge_counters = {} 
        self.liveness_state = {} 

    def reset(self):
        self.challenge_counters = {}
        self.liveness_state = {}

    def process_frame(self, frame, user_name="current_user", challenge="blink"):
        # 1. Check for Moire spoofing first
        is_screen, moire_score = self.moire_detector.analyze(frame)
        if is_screen:
            return False, f"SPOOF_SCREEN ({moire_score:.1f})"

        # Convert BGR to RGB since Mediapipe uses RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = self.face_landmarker.detect(mp_image)
        
        if user_name not in self.challenge_counters:
            self.challenge_counters[user_name] = 0
            self.liveness_state[user_name] = False

        if self.liveness_state[user_name]:
            return True, "VERIFIED"

        if results.face_landmarks:
            for face_landmarks in results.face_landmarks:
                h, w, _ = frame.shape
                
                # Helper to get pixel coords safely
                def get_pt(idx):
                    lm = face_landmarks[idx]
                    return (min(math.floor(lm.x * w), w - 1), min(math.floor(lm.y * h), h - 1))
                
                metric_val = 0.0
                target_met = False

                if challenge == "blink":
                    left_eye_points = [face_landmarks[p] for p in LEFT_EYE_INDICES]
                    right_eye_points = [face_landmarks[p] for p in RIGHT_EYE_INDICES]
                    left_ear = compute_ear(left_eye_points, w, h)
                    right_ear = compute_ear(right_eye_points, w, h)
                    avg_ear = (left_ear + right_ear) / 2.0
                    metric_val = avg_ear
                    
                    if avg_ear < self.ear_threshold:
                        self.challenge_counters[user_name] += 1
                    else:
                        if self.challenge_counters[user_name] >= self.frames_to_blink:
                            target_met = True
                        self.challenge_counters[user_name] = 0

                elif challenge == "smile":
                    left_corner = get_pt(61)
                    right_corner = get_pt(291)
                    left_cheek = get_pt(234)
                    right_cheek = get_pt(454)
                    
                    mouth_width = euclidean_distance(left_corner, right_corner)
                    face_width = euclidean_distance(left_cheek, right_cheek)
                    smile_ratio = round(mouth_width / (face_width + 1e-6), 3)
                    metric_val = smile_ratio
                    
                    if smile_ratio > 0.45: # Typically exceeds 0.45 during a smile
                        self.challenge_counters[user_name] += 1
                        if self.challenge_counters[user_name] >= 3: # Need consistency
                            target_met = True
                    else:
                        self.challenge_counters[user_name] = 0
                        
                elif challenge == "turn_left" or challenge == "turn_right":
                    nose_tip = get_pt(4)
                    left_cheek = get_pt(234)
                    right_cheek = get_pt(454)
                    
                    dist_left = euclidean_distance(nose_tip, left_cheek)
                    dist_right = euclidean_distance(nose_tip, right_cheek)
                    
                    ratio = round(dist_left / (dist_right + 1e-6), 2)
                    metric_val = ratio
                    
                    if challenge == "turn_left" and ratio < 0.6: # Nose closer to left cheek 
                        self.challenge_counters[user_name] += 1
                    elif challenge == "turn_right" and ratio > 1.6: # Nose closer to right cheek
                        self.challenge_counters[user_name] += 1
                    else:
                        self.challenge_counters[user_name] = 0
                        
                    if self.challenge_counters[user_name] >= 3:
                        target_met = True

                elif challenge == "look_up":
                    nose_tip = get_pt(4)
                    top_head = get_pt(10)
                    chin = get_pt(152)
                    
                    dist_top = euclidean_distance(nose_tip, top_head)
                    dist_bottom = euclidean_distance(nose_tip, chin)
                    
                    # When looking up, nose tip moves closer to top_head in 2D projection
                    ratio = round(dist_top / (dist_bottom + 1e-6), 2)
                    metric_val = ratio
                    
                    if ratio < 0.7: 
                        self.challenge_counters[user_name] += 1
                        if self.challenge_counters[user_name] >= 3:
                            target_met = True
                    else:
                        self.challenge_counters[user_name] = 0

                if target_met:
                    self.liveness_state[user_name] = True
                    return True, f"VERIFIED: {challenge}"
                    
                return False, f"Val: {metric_val}"
                
        return False, "NO_FACE"
