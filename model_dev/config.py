import numpy as np

# Change these to whatever sentences or words you want the AI to learn!
ACTIONS = np.array([
    'Hello',
    'Hi',
    'Bye',
    'Yes',
    'No',
    'Please',
    'Thanks',
    'Sorry',
    'Help',
    'Stop',
    'World',
    'Good',
    'Morning',
    'Night',
    'Eveining'
])

# Number of sequences (videos) per action
# Higher number = better accuracy, but takes longer to collect data
NO_SEQUENCES = 30 

# Number of frames per sequence
# 60 frames = ~2 seconds of recording (good for small sentences)
SEQUENCE_LENGTH = 30 

# The number of input features (40 lip landmarks * 2 coordinates)
INPUT_SIZE = 80
HIDDEN_SIZE = 64
NUM_CLASSES = len(ACTIONS)
