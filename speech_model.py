import time
import os
import json
import numpy as np
import cv2

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

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
        self.is_recording = False
        self.sequence_len = 20
        self.phrase_dict = {}
        
        if ONNX_AVAILABLE:
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'lip_phrase_baseline.onnx')
            config_path = os.path.join(os.path.dirname(__file__), 'config', 'phrases_50.json')
            
            if os.path.exists(model_path) and os.path.exists(config_path):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                        for phrase in config_data['phrases']:
                            self.phrase_dict[int(phrase['id'])] = phrase['display']
                            
                    self.ort_session = ort.InferenceSession(model_path)
                    self.use_dl = True
                    print("Successfully loaded Deep Learning Lip Reading Model (ONNX)!")
                except Exception as e:
                    print(f"Error loading ONNX model: {e}")
            else:
                print("ONNX model or config not found. Using heuristic model.")
        else:
            print("ONNXRuntime is not installed. Using heuristic model.")

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
        if self.use_dl and frame is not None:
            _, display_text, confidence, new_word = self._predict_dl(lip_data, frame)
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
            
    def _predict_dl(self, lip_data, frame):
        current_time = time.time()
        
        # Track MAR variance for silence detection
        self.mar_sequence.append(lip_data["mar"])
        if len(self.mar_sequence) > 10:
            self.mar_sequence.pop(0)
            
        mar = lip_data["mar"]
        hand_gesture = lip_data.get("hand_gesture", "None")
        display_text = self.current_sentence
        new_word = ""
        
        # Extract ROI
        roi = None
        if lip_data.get("lip_bbox") is not None:
            x1, y1, x2, y2 = lip_data["lip_bbox"]
            roi_bgr = frame[y1:y2+1, x1:x2+1]
            if roi_bgr.size > 0:
                roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
                roi = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)

        # Clear sentence after 4s of silence
        if not self.is_recording and current_time - self.last_prediction_time > 4.0:
            self.current_sentence = ""
            display_text = ""

        mar_variance = np.std(self.mar_sequence) if len(self.mar_sequence) == 10 else 0.0

        # Robust SAD: trigger only if mouth is moving (variance), mouth is reasonably open, and NO hand gesture
        if not self.is_recording:
            if mar_variance > 0.015 and mar > 0.10 and hand_gesture == "None":
                self.is_recording = True
                if roi is not None:
                    self.sequence = [roi]
        else:
            # If a hand gesture is detected during recording, we abort lip reading to avoid false positives
            if hand_gesture != "None":
                self.is_recording = False
                self.sequence = []
            elif roi is not None:
                self.sequence.append(roi)
                display_text = self.current_sentence
                
                if len(self.sequence) == self.sequence_len:
                    # We captured exactly sequence_len frames of a word! Predict it.
                    try:
                        # Prepare input tensor: [T, 64, 64]
                        seq_arr = np.stack(self.sequence, axis=0)
                        
                        # Normalize each clip identically to training
                        seq_arr = seq_arr.astype(np.float32) / 255.0
                        mean = seq_arr.mean()
                        std = seq_arr.std() + 1e-6
                        seq_arr = (seq_arr - mean) / std
                            
                        # Reshape to [B, T, 1, H, W] -> [1, 20, 1, 64, 64]
                        input_tensor = seq_arr.reshape(1, self.sequence_len, 1, 64, 64).astype(np.float32)
                        
                        ort_inputs = {self.ort_session.get_inputs()[0].name: input_tensor}
                        ort_outs = self.ort_session.run(None, ort_inputs)
                        logits = ort_outs[0][0]
                        
                        # Softmax
                        exp_logits = np.exp(logits - np.max(logits))
                        probs = exp_logits / np.sum(exp_logits)
                        predicted_idx = np.argmax(probs)
                        conf_val = probs[predicted_idx]
                        
                        # High confidence threshold to filter out random noise/head movements
                        if conf_val > 0.4:
                            predicted_word = self.phrase_dict.get(predicted_idx, "")
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
                    except Exception as e:
                        print(f"Inference error: {e}")
                        
                    self.is_recording = False
                    self.sequence = []
                    
        return "", display_text, conf_val if 'conf_val' in locals() else 0.0, new_word

    def _predict_heuristic(self, lip_data):
        mar = lip_data["mar"]
        width = lip_data["width"]
        hand_gesture = lip_data.get("hand_gesture", "None")
        
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
            
        # Block syllable tracking if hand gesture is active
        if hand_gesture == "None":
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
