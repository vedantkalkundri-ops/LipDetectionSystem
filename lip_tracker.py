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
        self.mp_drawing = mp.solutions.drawing_utils

    def process_frame(self, frame):
        """
        Processes a BGR frame, extracts lip landmarks, and calculates mouth aspect ratio and smile index.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        lip_data = {
            "landmarks": [],
            "mouth_open": False,
            "smile": False,
            "mar": 0.0, # Mouth Aspect Ratio
            "width": 0.0
        }

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                
                # Extract specific lip landmarks
                # Top lip: 13, Bottom lip: 14 (inner lips)
                # Left corner: 78, Right corner: 308
                pt_top = face_landmarks.landmark[13]
                pt_bottom = face_landmarks.landmark[14]
                pt_left = face_landmarks.landmark[78]
                pt_right = face_landmarks.landmark[308]
                
                # Convert to pixel coordinates
                y_top = int(pt_top.y * h)
                y_bottom = int(pt_bottom.y * h)
                x_left = int(pt_left.x * w)
                x_right = int(pt_right.x * w)
                
                # Calculate metrics
                mouth_height = abs(y_bottom - y_top)
                mouth_width = abs(x_right - x_left)
                
                if mouth_width > 0:
                    mar = mouth_height / mouth_width
                else:
                    mar = 0.0
                
                lip_data["mar"] = mar
                lip_data["width"] = mouth_width
                
                # Heuristics for basic gestures
                if mar > 0.15:
                    lip_data["mouth_open"] = True
                
                # Simple smile heuristic (mouth width relative to face width could be better, but we use an approximation)
                # Assuming standard face width mapping or just large mouth width
                # Alternatively, we could just send the mar and width to speech_model
                
                # Extract all lip landmarks for frontend rendering if needed
                for idx in self.mp_face_mesh.FACEMESH_LIPS:
                    start_idx = idx[0]
                    pt = face_landmarks.landmark[start_idx]
                    lip_data["landmarks"].append({"x": pt.x, "y": pt.y})

        return lip_data
