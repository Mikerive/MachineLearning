"""
Physics-Informed LSTM Network Architecture

This implementation uses a deep LSTM architecture specifically designed for time series forecasting
with physics-informed features. The architecture is structured to progressively learn more complex
patterns while maintaining numerical stability and preventing overfitting.

Architecture Overview:
---------------------
1. LSTM Layers (Progressive Feature Extraction):
   - LSTM1: Initial feature processing (size: input → hidden_size)
     Purpose: Capture basic temporal patterns and local dependencies
   
   - LSTM2: Intermediate processing (size: hidden_size → hidden_size*2)
     Purpose: Learn more complex feature combinations and medium-term patterns
   
   - LSTM3: Deep feature extraction (size: hidden_size*2 → hidden_size*4)
     Purpose: Capture long-term dependencies and complex interactions

2. Regularization Components:
   - Dropout layers (20% rate)
     Purpose: Prevent overfitting and improve generalization
   
   - Batch Normalization
     Purpose: Stabilize training, reduce internal covariate shift, allow higher learning rates

3. Fully Connected Layers:
   - FC1: (hidden_size*4 → hidden_size*2) with ReLU
     Purpose: Combine extracted features and reduce dimensionality
   
   - FC2: (hidden_size*2 → hidden_size) with ReLU
     Purpose: Further feature abstraction and pattern recognition
   
   - FC3: (hidden_size → 1)
     Purpose: Final regression output

Training Considerations:
-----------------------
- Uses progressive hidden size increase to prevent information bottleneck
- Employs multiple regularization techniques to handle complex feature interactions
- Implements modern initialization strategies for stable gradient flow
- Batch size should be tuned based on the sequence length and feature count
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsInformedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.2):
        """
        Initialize the Physics-Informed LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Base size for hidden layers (progressively increases)
            num_layers: Number of LSTM layers (currently fixed at 3)
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        # Multi-layer LSTM with increasing size for progressive feature extraction
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        
        self.lstm2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size*2,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        
        self.lstm3 = nn.LSTM(
            input_size=hidden_size*2,
            hidden_size=hidden_size*4,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        
        # Dropout layers for regularization
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        # Batch normalization for training stability
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size*2)
        self.bn3 = nn.BatchNorm1d(hidden_size*4)
        
        # Fully connected layers with decreasing sizes for feature combination
        self.fc1 = nn.Linear(hidden_size*4, hidden_size*2)
        self.fc2 = nn.Linear(hidden_size*2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights using different strategies for different layer types:
        - LSTM: Xavier uniform for input weights, Orthogonal for hidden weights
        - FC: Kaiming normal for better ReLU activation
        """
        for lstm in [self.lstm1, self.lstm2, self.lstm3]:
            for name, param in lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        
        # Initialize FC layers with Kaiming initialization
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        
        # Initialize biases to zero
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        self.fc3.bias.data.fill_(0)

    def forward(self, x):
        """
        Forward pass through the network.
        Implements progressive feature extraction with regularization at each step.
        """
        # First LSTM layer - Basic temporal pattern extraction
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)
        last_out1 = lstm1_out[:, -1, :]
        last_out1 = self.bn1(last_out1)
        
        # Second LSTM layer - Intermediate pattern processing
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.dropout2(lstm2_out)
        last_out2 = lstm2_out[:, -1, :]
        last_out2 = self.bn2(last_out2)
        
        # Third LSTM layer - Complex pattern extraction
        lstm3_out, _ = self.lstm3(lstm2_out)
        lstm3_out = self.dropout3(lstm3_out)
        last_out3 = lstm3_out[:, -1, :]
        last_out3 = self.bn3(last_out3)
        
        # Fully connected layers with ReLU activation
        fc1_out = F.relu(self.fc1(last_out3))
        fc1_out = self.dropout1(fc1_out)
        
        fc2_out = F.relu(self.fc2(fc1_out))
        fc2_out = self.dropout2(fc2_out)
        
        # Final output layer (no activation for regression)
        output = self.fc3(fc2_out)
        
        return output
