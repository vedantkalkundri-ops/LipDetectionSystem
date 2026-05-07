import cv2
import numpy as np
from lip_tracker import LipTracker
from speech_model import SpeechModel

def test():
    tracker = LipTracker()
    model = SpeechModel()
    
    # Create a dummy image
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    try:
        lip_data = tracker.process_frame(frame)
        prediction = model.predict(lip_data, language="en")
        print("Success! Gesture:", prediction["gesture"], "Text:", prediction["predicted_text"])
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
