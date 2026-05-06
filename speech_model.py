import time

class SpeechModel:
    def __init__(self):
        self.history = []
        self.last_predicted_word = ""
        self.last_prediction_time = 0
        
        # State tracking for syllable counting
        self.is_open = False
        self.syllable_count = 0
        self.last_syllable_time = time.time()
        
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
        
        return gesture, display_text

