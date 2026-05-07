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
    'Wait',
    'Come',
    'Go',
    'Open',
    'Close',
    'Eat',
    'Drink',
    'Water',
    'Need',
    'Want',
    'Give',
    'Take',
    'Where',
    'What',
    'Who',
    'Why',
    'When',
    'How',
    'You',
    'Me',
    'I',
    'Fine',
    'Good',
    'Bad',
    'Okay',
    'Name',
    'Home',
    'Work',
    'Sleep',
    'Call',
    'World'
])

# Number of sequences (videos) per action
# Higher number = better accuracy, but takes longer to collect data
NO_SEQUENCES = 10 

# Number of frames per sequence
# 60 frames = ~2 seconds of recording (good for small sentences)
SEQUENCE_LENGTH = 60 

# The number of input features (40 lip landmarks * 2 coordinates)
INPUT_SIZE = 80
HIDDEN_SIZE = 64
NUM_CLASSES = len(ACTIONS)
