import cv2
import mediapipe as mp
import numpy as np
import os
import time

from config import ACTIONS, NO_SEQUENCES, SEQUENCE_LENGTH

# Directory for dataset
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset')

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Extract unique lip indices from FACEMESH_LIPS
LIP_INDICES = list(set([idx for tup in mp_face_mesh.FACEMESH_LIPS for idx in tup]))
LIP_INDICES.sort()
NUM_FEATURES = len(LIP_INDICES) * 2 # (x, y) for each point

def extract_keypoints(results):
    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]
        lip_points = []
        for idx in LIP_INDICES:
            pt = face.landmark[idx]
            lip_points.extend([pt.x, pt.y])
        return np.array(lip_points)
    else:
        # If no face, return zeros
        return np.zeros(NUM_FEATURES)

def main():
    # Setup folders
    for action in ACTIONS:
        for sequence in range(NO_SEQUENCES):
            try: 
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass

    print("Opening Camera...")
    cam_idx_str = input("Enter camera index (0 for default laptop cam, 1 or 2 for DroidCam/Iriun mobile cam) [0]: ")
    cam_idx = int(cam_idx_str) if cam_idx_str.strip().isdigit() else 0
    cap = cv2.VideoCapture(cam_idx)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {cam_idx}. Try a different index.")
        return
    
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
        
        for action in ACTIONS:
            print(f"\n======================================")
            print(f"Get ready to record '{action}'")
            print(f"======================================")
            time.sleep(2) # Give user time to read
            
            sequence = 0
            while sequence < NO_SEQUENCES:
                # Check if this sequence is already fully recorded (Resume feature)
                seq_dir = os.path.join(DATA_PATH, action, str(sequence))
                if os.path.exists(seq_dir) and len(os.listdir(seq_dir)) == SEQUENCE_LENGTH:
                    sequence += 1
                    continue
                    
                paused = False
                frame_num = 0
                
                while frame_num < SEQUENCE_LENGTH:
                    ret, frame = cap.read()
                    if not ret:
                        # Camera drop - wait and retry without advancing frame_num
                        cv2.waitKey(100)
                        continue
                    
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = face_mesh.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                    # Draw landmarks for feedback
                    if results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=results.multi_face_landmarks[0],
                            connections=mp_face_mesh.FACEMESH_LIPS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                        )
                    
                    # Display collection info
                    if frame_num == 0: 
                        cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, f'Say "{action}"! (Video {sequence+1}/{NO_SEQUENCES})', (15,30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.imshow('Data Collection', image)
                        cv2.waitKey(2000) # Wait 2 seconds before starting the sequence
                    else: 
                        cv2.putText(image, f'Say "{action}"! (Video {sequence+1}/{NO_SEQUENCES})', (15,30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.imshow('Data Collection', image)
                        
                    # Export keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    # Break gracefully or pause
                    key = cv2.waitKey(10) & 0xFF
                    if key == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        print("\nData collection stopped by user!")
                        return
                    elif key == ord('p'):
                        paused = True
                        break # Break the frame loop
                        
                    frame_num += 1
                
                if paused:
                    print(f"PAUSED at Video {sequence+1}. Press 'p' in the window to resume.")
                    while True:
                        ret, frame = cap.read()
                        if not ret: continue
                        cv2.putText(frame, 'PAUSED', (200,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4, cv2.LINE_AA)
                        cv2.putText(frame, 'Press "p" to resume', (150,250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                        cv2.imshow('Data Collection', frame)
                        if cv2.waitKey(10) & 0xFF == ord('p'):
                            break
                    print("RESUMING...")
                    # Give a brief countdown
                    ret, frame = cap.read()
                    cv2.putText(frame, 'GET READY...', (150,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.imshow('Data Collection', frame)
                    cv2.waitKey(2000)
                    continue # Restart the current sequence
                    
                sequence += 1
                    
    cap.release()
    cv2.destroyAllWindows()
    print("\nData collection finished!")

if __name__ == '__main__':
    main()
