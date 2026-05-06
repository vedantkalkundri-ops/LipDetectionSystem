import time

class SpeechModel:
    def __init__(self):
        # We will use a simple heuristic model for demonstration
        # It interprets consecutive frames of lip features into words
        self.history = []
        self.last_predicted_word = ""
        self.last_prediction_time = 0
        
    def predict(self, lip_data):
        """
        Takes lip features from a single frame and updates the state.
        Returns the current gesture and predicted text.
        """
        if not lip_data or lip_data["width"] == 0:
            return "No Face", ""
            
        mar = lip_data["mar"]
        width = lip_data["width"]
        
        current_time = time.time()
        
        # Determine current gesture
        gesture = "Neutral"
        if mar > 0.20:
            gesture = "Mouth Open (Vowel/Ah)"
        elif mar > 0.10:
            gesture = "Speaking..."
        elif width > 120: # Arbitrary threshold, depends on camera distance
            gesture = "Smile / Wide"
        else:
            gesture = "Mouth Closed"
            
        self.history.append({"mar": mar, "time": current_time, "gesture": gesture})
        
        # Keep only last 2 seconds of history
        self.history = [h for h in self.history if current_time - h["time"] < 2.0]
        
        # Simple heuristic pattern matching
        # If we see "Mouth Open" recently followed by "Mouth Closed", we can trigger a word
        # In a real model, this would be an LSTM or Transformer predicting characters/phonemes
        
        predicted_text = ""
        
        open_count = sum(1 for h in self.history if h["gesture"] == "Mouth Open (Vowel/Ah)")
        
        if current_time - self.last_prediction_time > 3.0:
            if open_count > 10:
                predicted_text = "Hello"
                self.last_predicted_word = predicted_text
                self.last_prediction_time = current_time
            elif open_count > 5:
                predicted_text = "Yes"
                self.last_predicted_word = predicted_text
                self.last_prediction_time = current_time
                
        # To make the UI dynamic, we clear the predicted word after some time
        display_text = self.last_predicted_word if current_time - self.last_prediction_time < 3.0 else ""
        
        return gesture, display_text
