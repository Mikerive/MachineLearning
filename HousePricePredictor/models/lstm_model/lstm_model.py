import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch

from .components.lstm_network import PhysicsInformedLSTM
from .components.training_monitor import TrainingMonitor

class PhysicsInformedLSTMModel(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        model_name='lstm',
        sequence_length=24,  # Increased from 12 to capture longer patterns
        hidden_size=128,    # Increased base hidden size
        epochs=50,          # More epochs for deeper network
        batch_size=64,      # Increased for better gradient estimates
        learning_rate=0.0005,  # Reduced for stability
        patience=10,        # Increased patience for deeper network
        min_delta=0.0001,   # Reduced for finer convergence
        weight_decay=0.001  # Reduced weight decay
    ):
        self.model_name = model_name
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.min_delta = min_delta
        self.weight_decay = weight_decay
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.monitor = TrainingMonitor(model_name)

    def _reshape_for_model(self, X: np.ndarray) -> np.ndarray:
        """
        Reshape standard (n_samples, n_features) input into LSTM sequences.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Array of shape (n_samples - sequence_length, sequence_length, n_features)
        """
        sequences = []
        for i in range(len(X) - self.sequence_length):
            seq = X[i:i+self.sequence_length]
            sequences.append(seq)
        return np.array(sequences)

    def _inverse_reshape(self, X: np.ndarray) -> np.ndarray:
        """
        Convert LSTM sequences back to standard format.
        Takes last timestep from each sequence.
        
        Args:
            X: Array of shape (n_samples, sequence_length, n_features)
            
        Returns:
            Array of shape (n_samples, n_features)
        """
        return X[:, -1, :]

    def _prepare_sequences(self, X, y=None):
        """
        Prepare sequences for LSTM model.
        
        Args:
            X: Features array of shape (n_samples, n_features)
            y: Target array of shape (n_samples,)
            
        Returns:
            If y is None:
                sequences: Array of shape (n_samples - sequence_length, sequence_length, n_features)
            If y is not None:
                sequences: Array of shape (n_samples - sequence_length, sequence_length, n_features)
                targets: Array of shape (n_samples - sequence_length,)
        """
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Note: Features are already scaled in preprocessing
        X_scaled = X
        
        X_seq = self._reshape_for_model(X_scaled)
        
        if y is not None:
            y = np.array(y)
            y_seq = y[self.sequence_length:]
            return X_seq, y_seq
        return X_seq

    def fit(self, X, y):
        """
        Fit the LSTM model.
        
        Args:
            X: Features array of shape (n_samples, n_features)
            y: Target array of shape (n_samples,)
        """
        X = np.array(X)
        y = np.array(y)
        
        # Validate input shapes and types
        print("\nValidating Input Data:")
        print("----------------------")
        print(f"Input Features Shape: {X.shape}")
        print(f"Target Shape: {y.shape}")
        
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError(f"Features must be numeric. Got dtype: {X.dtype}")
        if not np.issubdtype(y.dtype, np.number):
            raise ValueError(f"Target must be numeric. Got dtype: {y.dtype}")
        
        assert X.shape[0] == y.shape[0], f"X and y must have same number of samples. Got X: {X.shape[0]}, y: {y.shape[0]}"
        assert len(X.shape) == 2, f"X must be 2D array. Got shape: {X.shape}"
        assert len(y.shape) == 1, f"y must be 1D array. Got shape: {y.shape}"
        
        # Create sequences
        X_seq, y_seq = self._prepare_sequences(X, y)
        
        # Validate sequence shapes
        print("\nSequence Shapes:")
        print("---------------")
        print(f"X sequences: {X_seq.shape} (samples, sequence_length, features)")
        print(f"y sequences: {y_seq.shape} (samples,)")
        
        expected_seq_shape = (X.shape[0] - self.sequence_length, self.sequence_length, X.shape[1])
        assert X_seq.shape == expected_seq_shape, \
            f"Invalid sequence shape. Expected {expected_seq_shape}, got {X_seq.shape}"
        
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_seq, dtype=torch.float32).to(self.device)
        
        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            drop_last=True
        )
        
        # Initialize LSTM model
        input_size = X_seq.shape[2]  # number of features
        self.lstm = PhysicsInformedLSTM(
            input_size=input_size, 
            hidden_size=self.hidden_size
        ).to(self.device)
        
        print(f"\nModel Configuration:")
        print("-------------------")
        print(f"Input Size: {input_size}")
        print(f"Hidden Size: {self.hidden_size}")
        print(f"Sequence Length: {self.sequence_length}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Device: {self.device}")
        
        # Print sequence shapes for debugging
        print(f"\nSequence shapes:")
        print(f"X shape: {X_tensor.shape} (samples, sequence_length, features)")
        print(f"y shape: {y_tensor.shape} (samples,)")
        
        optimizer = torch.optim.Adam(self.lstm.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, patience=2, verbose=True, min_lr=1e-6, threshold=self.min_delta
        )
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.lstm.train()
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_X, batch_y in dataloader:
                # Zero gradients for each batch
                optimizer.zero_grad()
                
                # Forward pass
                predictions = self.lstm(batch_X)
                mse_loss = torch.mean((predictions.squeeze() - batch_y)**2)
                loss = mse_loss
                
                # Backward pass and optimization
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.lstm.parameters(), max_norm=1.0)  # Add gradient clipping
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / batch_count
            
            # Update learning rate scheduler with epoch loss
            scheduler.step(avg_epoch_loss)
            
            # Record loss
            self.monitor.update(avg_epoch_loss, epoch, optimizer.param_groups[0]['lr'])
            
            # Early stopping check
            if avg_epoch_loss < best_loss - self.min_delta:
                best_loss = avg_epoch_loss
                best_model_state = self.lstm.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_epoch_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Load best model
        self.lstm.load_state_dict(best_model_state)
        return self

    def predict(self, X):
        self.lstm.eval()
        X_seq = self._prepare_sequences(X)
        X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            predictions = self.lstm(X_tensor)
        
        return predictions.cpu().numpy().flatten()
