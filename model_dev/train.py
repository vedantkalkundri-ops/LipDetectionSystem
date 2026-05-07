import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

from config import ACTIONS, NO_SEQUENCES, SEQUENCE_LENGTH, INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset')

class LipDataset(Dataset):
    def __init__(self):
        self.sequences = []
        self.labels = []
        
        label_map = {label:num for num, label in enumerate(ACTIONS)}
        
        for action in ACTIONS:
            action_path = os.path.join(DATA_PATH, action)
            if not os.path.exists(action_path): continue
            
            # Dynamically load all available sequences for this action
            for seq_str in os.listdir(action_path):
                try:
                    sequence = int(seq_str)
                except ValueError:
                    continue
                    
                window = []
                for frame_num in range(SEQUENCE_LENGTH):
                    try:
                        res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                        window.append(res)
                    except:
                        pass
                if len(window) == SEQUENCE_LENGTH:
                    self.sequences.append(window)
                    self.labels.append(label_map[action])
                    
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        # Convert sequence to numpy array for processing
        seq = np.array(self.sequences[idx], dtype=np.float32)
        
        # --- DATA AUGMENTATION (Applied randomly 50% of the time) ---
        # 1. Temporal Shifting (Roll frames forward/backward slightly)
        if torch.rand(1).item() < 0.5:
            shift = np.random.randint(-3, 4) # Shift between -3 and 3 frames
            seq = np.roll(seq, shift, axis=0)
            
        # 2. Spatial Noise (Slight jitter on coordinates)
        if torch.rand(1).item() < 0.5:
            noise = np.random.normal(0, 0.01, seq.shape).astype(np.float32)
            seq += noise
        # ------------------------------------------------------------
        
        # Normalize each frame in the sequence
        for i in range(len(seq)):
            xs = seq[i, 0::2]
            ys = seq[i, 1::2]
            
            # Center points
            seq[i, 0::2] = xs - np.mean(xs)
            seq[i, 1::2] = ys - np.mean(ys)
            
            # Scale by bounding box
            max_val = np.max(np.abs(seq[i]))
            if max_val > 0:
                seq[i] /= max_val
                
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

class LipLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LipLSTM, self).__init__()
        # Use 2 LSTM layers with dropout between them
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.5)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # Get last time step
        out = out[:, -1, :] 
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def train_model():
    print("Loading dataset...")
    dataset = LipDataset()
    if len(dataset) == 0:
        print("Dataset not found! Please run collect_data.py first.")
        return
        
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LipLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 150
    print(f"Training on {device} for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        
        for sequences, labels in dataloader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}, Accuracy: {100 * correct / total:.2f}%')
            
    # Save model
    save_path = os.path.join(os.path.dirname(__file__), 'lip_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    train_model()
