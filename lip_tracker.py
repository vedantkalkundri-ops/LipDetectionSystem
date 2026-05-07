import cv2
import mediapipe as mp
import numpy as np

class LipTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True
        )
        
        # Robust Hand Tracking instead of Task API
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.last_mar = 0.0

    def process_frame(self, frame):
        """
        Processes a BGR frame, extracts lip landmarks, calculates metrics, and recognizes gestures.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process Face
        results = self.face_mesh.process(rgb_frame)
        
        # Process Hands for robust gestures
        hand_results = self.hands.process(rgb_frame)
        hand_gesture = "None"
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Get y-coordinates of fingertips and lower joints
                tips = [
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP].y
                ]
                pips = [
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP].y,
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP].y
                ]
                
                # Check if fingers are extended (tip is higher than pip, meaning lower y value)
                fingers_up = [tips[i] < pips[i] for i in range(1, 5)]
                thumb_up = tips[0] < pips[0]
                
                if all(fingers_up) and thumb_up:
                    hand_gesture = "Open_Palm"
                elif thumb_up and not any(fingers_up):
                    # To ensure it's a real thumbs up, thumb tip should be significantly higher than index pip
                    if tips[0] < pips[1] - 0.05:
                        hand_gesture = "Thumb_Up"
                    else:
                        hand_gesture = "Closed_Fist"
                elif not thumb_up and not any(fingers_up):
                    hand_gesture = "Closed_Fist"
                elif fingers_up[0] and fingers_up[1] and not fingers_up[2] and not fingers_up[3]:
                    hand_gesture = "Victory"
        
        lip_data = {
            "landmarks": [],
            "mouth_open": False,
            "smile": False,
            "mar": 0.0,
            "width": 0.0,
            "spread": 0.0,
            "openness_velocity": 0.0,
            "lip_bbox": None,
            "hand_gesture": hand_gesture,
            "facial_gesture": "Neutral"
        }

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                
                pt_top = face_landmarks.landmark[13]
                pt_bottom = face_landmarks.landmark[14]
                pt_left = face_landmarks.landmark[78]
                pt_right = face_landmarks.landmark[308]
                
                y_top = int(pt_top.y * h)
                y_bottom = int(pt_bottom.y * h)
                x_left = int(pt_left.x * w)
                x_right = int(pt_right.x * w)
                
                mouth_height = abs(y_bottom - y_top)
                mouth_width = abs(x_right - x_left)
                
                if mouth_width > 0:
                    mar = mouth_height / mouth_width
                else:
                    mar = 0.0
                
                spread = min(1.0, mouth_width / float(max(w, 1)))
                
                lip_data["mar"] = mar
                lip_data["width"] = mouth_width
                lip_data["spread"] = spread
                lip_data["openness_velocity"] = mar - self.last_mar
                self.last_mar = mar

                # Extract all unique lip landmarks
                lip_indices = list(set([idx for tup in self.mp_face_mesh.FACEMESH_LIPS for idx in tup]))
                lip_indices.sort()
                for idx in lip_indices:
                    pt = face_landmarks.landmark[idx]
                    lip_data["landmarks"].append({"x": pt.x, "y": pt.y})
                    
                # Facial Gestures Heuristics
                if mar > 0.15:
                    lip_data["facial_gesture"] = "Mouth Open / Surprise"
                    lip_data["mouth_open"] = True
                elif spread > 0.18 and mar < 0.10: # Use relative spread instead of absolute width
                    lip_data["facial_gesture"] = "Smile"
                    lip_data["smile"] = True
                else:
                    # Additional Facial Gestures (Sad, Angry)
                    # Calculate face height for normalization
                    pt_chin = face_landmarks.landmark[152]
                    pt_forehead = face_landmarks.landmark[10]
                    face_height = abs(pt_chin.y - pt_forehead.y)
                    
                    if face_height > 0:
                        # Sad (mouth corners pulled down relative to center)
                        pt_left_corner = face_landmarks.landmark[61]
                        pt_right_corner = face_landmarks.landmark[291]
                        pt_top_lip = face_landmarks.landmark[13]
                        pt_bottom_lip = face_landmarks.landmark[14]
                        
                        y_center = (pt_top_lip.y + pt_bottom_lip.y) / 2.0
                        y_corners = (pt_left_corner.y + pt_right_corner.y) / 2.0
                        
                        frown_metric = y_corners - y_center # Positive if corners are lower
                        frown_ratio = frown_metric / face_height
                        
                        # Angry (inner eyebrows pulled down close to the eyes)
                        # Eye tops
                        y_left_eye = face_landmarks.landmark[159].y
                        y_right_eye = face_landmarks.landmark[386].y
                        # Eyebrow inner bottoms
                        y_left_eyebrow = face_landmarks.landmark[52].y
                        y_right_eyebrow = face_landmarks.landmark[282].y
                        
                        scowl_left = y_left_eye - y_left_eyebrow
                        scowl_right = y_right_eye - y_right_eyebrow
                        scowl_metric = (scowl_left + scowl_right) / 2.0
                        scowl_ratio = scowl_metric / face_height
                        
                        if frown_ratio > 0.012:
                            lip_data["facial_gesture"] = "Sad"
                        elif scowl_ratio < 0.038: # Eyebrows are very close to eyes
                            lip_data["facial_gesture"] = "Angry"

        return lip_data
