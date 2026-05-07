import time
import os
import numpy as np

# Try to import torch and the model, but fall back gracefully
try:
    import torch
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), 'model_dev'))
    from config import INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES, ACTIONS, SEQUENCE_LENGTH
    from train import LipLSTM
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class SpeechModel:
    def __init__(self):
        self.last_predicted_word = ""
        self.last_prediction_time = 0
        self.current_sentence = ""
        self.mar_sequence = []
        
        # Heuristic state tracking for syllable counting
        self.is_open = False
        self.syllable_count = 0
        self.last_syllable_time = time.time()
        
        # Deep Learning state
        self.use_dl = False
        self.sequence = []
        
        if TORCH_AVAILABLE:
            model_path = os.path.join(os.path.dirname(__file__), 'model_dev', 'lip_model.pth')
            if os.path.exists(model_path):
                try:
                    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    self.model = LipLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(self.device)
                    self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                    self.model.eval()
                    self.use_dl = True
                    print("Successfully loaded Deep Learning Lip Reading Model!")
                except Exception as e:
                    print(f"Error loading model: {e}")
            else:
                print("lip_model.pth not found. Using heuristic model.")

    def predict(self, lip_data, language="en", frame=None):
        """
        Takes lip features from a single frame and updates the state.
        Returns a dictionary with prediction results.
        """
        if not lip_data or lip_data["width"] == 0:
            return {
                "gesture": "", 
                "predicted_text": "",
                "partial_text": "",
                "confidence": 0.0,
                "phrase_id": "",
                "is_final": False,
                "new_word": ""
            }
            
        new_word = ""
        if self.use_dl:
            _, display_text, confidence, new_word = self._predict_dl(lip_data)
        else:
            _, display_text, confidence = self._predict_heuristic(lip_data)
            
        return {
            "gesture": "",
            "predicted_text": display_text,
            "partial_text": "..." if "Buffering" in display_text or display_text == "..." else "",
            "confidence": confidence,
            "phrase_id": "",
            "is_final": display_text != "" and "Buffering" not in display_text and display_text != "...",
            "new_word": new_word
        }
            
    def _predict_dl(self, lip_data):
        current_time = time.time()
        
        # Track MAR variance for silence detection
        self.mar_sequence.append(lip_data["mar"])
        self.mar_sequence = self.mar_sequence[-SEQUENCE_LENGTH:]
        
        # Extract features directly from landmarks (40 points * 2)
        lip_points = []
        for pt in lip_data["landmarks"]:
            lip_points.extend([pt["x"], pt["y"]])
            
        # Normalize the incoming frame just like during training
        xs = lip_points[0::2]
        ys = lip_points[1::2]
        mean_x = np.mean(xs)
        mean_y = np.mean(ys)
        
        normalized_points = []
        mar = lip_data["mar"]
        display_text = self.current_sentence
        new_word = ""

        # Extract landmarks
        landmarks = lip_data["landmarks"]
        xs = [pt['x'] for pt in landmarks]
        ys = [pt['y'] for pt in landmarks]
        
        mean_x, mean_y = np.mean(xs), np.mean(ys)
        norm_x = [x - mean_x for x in xs]
        norm_y = [y - mean_y for y in ys]
        
        max_val = max(np.max(np.abs(norm_x)), np.max(np.abs(norm_y)), 1e-5)
        norm_x = [x / max_val for x in norm_x]
        norm_y = [y / max_val for y in norm_y]
        
        normalized_points = []
        for nx, ny in zip(norm_x, norm_y):
            normalized_points.extend([nx, ny])
            
        # Clear sentence after 4s of silence
        if not getattr(self, 'is_recording', False) and current_time - self.last_prediction_time > 4.0:
            self.current_sentence = ""
            display_text = ""

        # Trigger-based Speech Activity Detection
        if not getattr(self, 'is_recording', False):
            if mar > 0.12: # Mouth opened significantly, start recording a word!
                self.is_recording = True
                self.sequence = [normalized_points]
        else:
            self.sequence.append(normalized_points)
            display_text = self.current_sentence # Do not show 'Listening...', keep the sentence
            
            if len(self.sequence) == SEQUENCE_LENGTH:
                # We captured exactly 60 frames of a word! Predict it.
                with torch.no_grad():
                    input_tensor = torch.tensor([self.sequence], dtype=torch.float32).to(self.device)
                    res = self.model(input_tensor)
                    confidence, predicted_idx = torch.max(res, 1)
                    
                    predicted_word = ACTIONS[predicted_idx.item()]
                    conf_val = float(confidence[0].item())
                    
                    # Only accept high confidence predictions to stop hallucinations
                    if conf_val > 0.6: 
                        new_word = predicted_word
                        if self.current_sentence == "":
                            self.current_sentence = predicted_word
                        else:
                            if predicted_word != self.last_predicted_word:
                                self.current_sentence += " " + predicted_word
                            else:
                                self.current_sentence += " " + predicted_word
                            
                        self.last_predicted_word = predicted_word
                        self.last_prediction_time = current_time
                        
                    display_text = self.current_sentence
                    self.is_recording = False
                    self.sequence = []
                    
        return "", display_text, conf_val if 'conf_val' in locals() else 0.0, new_word

    def _predict_heuristic(self, lip_data):
        mar = lip_data["mar"]
        width = lip_data["width"]
        
        current_time = time.time()
        
        # Determine current gesture
        gesture = "Neutral"
        if mar > 0.15:
            gesture = "Mouth Open"
        elif mar > 0.10:
            gesture = "Speaking..."
        elif width > 120:
            gesture = "Smile / Wide"
        else:
            gesture = "Mouth Closed"
            
        # Detect state transitions (Syllable tracking)
        if mar > 0.15 and not self.is_open:
            self.is_open = True
            
        elif mar <= 0.10 and self.is_open:
            # Transitioned from Open -> Closed (One syllable)
            self.is_open = False
            self.syllable_count += 1
            self.last_syllable_time = current_time
            
        predicted_text = ""
        
        # If user stops speaking for 1.5 seconds, evaluate the syllables
        if self.syllable_count > 0 and (current_time - self.last_syllable_time > 1.5):
            if self.syllable_count == 1:
                predicted_text = "Yes"
            elif self.syllable_count == 2:
                predicted_text = "Hello"
            elif self.syllable_count == 3:
                predicted_text = "How are you?"
            elif self.syllable_count >= 4:
                predicted_text = "I am fine"
                
            self.last_predicted_word = predicted_text
            self.last_prediction_time = current_time
            self.syllable_count = 0 # Reset
            
        # To make the UI dynamic, we clear the predicted word after some time
        display_text = self.last_predicted_word if current_time - self.last_prediction_time < 3.0 else ""
        
        # If currently speaking, show a typing indicator like effect
        if self.syllable_count > 0 and display_text == "":
            display_text = "..."
        
        return gesture, display_text, 0.5
