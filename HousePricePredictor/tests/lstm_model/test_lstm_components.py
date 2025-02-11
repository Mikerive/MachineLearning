"""
Tests for LSTM model components.
"""

import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import sys
import tempfile
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import os
import json

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from models.lstm_model.components.lstm_network import PhysicsInformedLSTM
from models.lstm_model.components.training_monitor import TrainingMonitor
from models.lstm_model.components.performance_analyzer import ModelPerformanceAnalyzer
from models.lstm_model.components.model_evaluation import ModelEvaluator, PerformanceMetrics, VisualizationTools

class TestPhysicsInformedLSTM(unittest.TestCase):
    """Test cases for LSTM network architecture."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_size = 10
        self.hidden_size = 64
        self.network = PhysicsInformedLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=3,
            dropout=0.2
        )
    
    def test_network_build(self):
        """Test network architecture initialization."""
        # Check layer types
        self.assertIsInstance(self.network.lstm1, nn.LSTM)
        self.assertIsInstance(self.network.lstm2, nn.LSTM)
        self.assertIsInstance(self.network.lstm3, nn.LSTM)
        self.assertIsInstance(self.network.fc1, nn.Linear)
        self.assertIsInstance(self.network.fc2, nn.Linear)
        self.assertIsInstance(self.network.fc3, nn.Linear)
    
    def test_forward_pass(self):
        """Test forward pass through the network."""
        batch_size = 32
        seq_length = 30
        
        # Generate sample input
        x = torch.randn(batch_size, seq_length, self.input_size)
        
        # Forward pass
        output = self.network(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, 1))
        self.assertTrue(torch.all(torch.isfinite(output)))
    
    def test_weight_initialization(self):
        """Test weight initialization."""
        # Check LSTM weight initialization
        for lstm in [self.network.lstm1, self.network.lstm2, self.network.lstm3]:
            for name, param in lstm.named_parameters():
                if 'weight_ih' in name or 'weight_hh' in name:
                    # Weights should be initialized and finite
                    self.assertTrue(torch.all(torch.isfinite(param)))
                    # Weights should not be zero
                    self.assertFalse(torch.all(param == 0))
        
        # Check FC layer initialization
        for fc in [self.network.fc1, self.network.fc2, self.network.fc3]:
            # Weights should be initialized and finite
            self.assertTrue(torch.all(torch.isfinite(fc.weight)))
            # Biases should be zero
            self.assertTrue(torch.all(fc.bias == 0))

class TestTrainingMonitor(unittest.TestCase):
    """Test cases for training monitoring functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test logs
        self.test_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.test_dir)  # Change to temp dir for test
        
        # Create training logs directory
        self.model_dir = os.path.join(self.test_dir, "training_logs", "TestModel_test")
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Patch the TrainingMonitor to use a fixed directory
        def fixed_init(self, model_name):
            if not hasattr(self, 'initialized'):
                self.model_name = model_name
                self.timestamp = "test"
                self.save_dir = os.path.join(os.getcwd(), "training_logs", f"{model_name}_{self.timestamp}")
                os.makedirs(self.save_dir, exist_ok=True)
                
                self.fig, self.ax = plt.subplots(figsize=(10, 6))
                self.fig.canvas.manager.set_window_title(f'Training Monitor - {model_name}')
                self.initialized = True
                self.loss_history = []
        
        # Apply the patch
        self._original_init = TrainingMonitor.__init__
        TrainingMonitor.__init__ = fixed_init
        
        self.monitor = TrainingMonitor("TestModel")
    
    def test_singleton_pattern(self):
        """Test that TrainingMonitor follows singleton pattern."""
        monitor2 = TrainingMonitor("AnotherModel")
        self.assertEqual(self.monitor, monitor2)
        self.assertEqual(self.monitor.model_name, monitor2.model_name)
    
    def test_loss_tracking(self):
        """Test loss history tracking."""
        # Update with some test losses
        self.monitor.update(mse_loss=0.5, phys_loss=0.3, lr=0.001)
        self.monitor.update(mse_loss=0.4, phys_loss=0.25, lr=0.001)
        
        # Check loss history
        self.assertEqual(len(self.monitor.loss_history), 2)
        self.assertEqual(self.monitor.loss_history[0]['mse'], 0.5)
        self.assertEqual(self.monitor.loss_history[0]['physics'], 0.3)
        self.assertEqual(self.monitor.loss_history[1]['mse'], 0.4)
        self.assertEqual(self.monitor.loss_history[1]['physics'], 0.25)
    
    def test_save_functionality(self):
        """Test saving of training logs."""
        # Update with test data
        self.monitor.update(mse_loss=0.5, phys_loss=0.3, lr=0.001)
        
        # Check if files are created
        log_path = os.path.join(self.monitor.save_dir, 'training_log.json')
        plot_path = os.path.join(self.monitor.save_dir, 'loss_plot.png')
        
        self.assertTrue(os.path.exists(log_path))
        self.assertTrue(os.path.exists(plot_path))
        
        # Check log content
        with open(log_path, 'r') as f:
            log_data = json.load(f)
            self.assertEqual(log_data['model_name'], "TestModel")
            self.assertTrue('history' in log_data)
            self.assertEqual(len(log_data['history']), 1)
    
    def tearDown(self):
        """Clean up after tests."""
        plt.close('all')  # Close all figures
        
        # Restore original init
        TrainingMonitor.__init__ = self._original_init
        
        # Change back to original directory
        os.chdir(self.original_dir)
        
        # Clean up created files
        if hasattr(self, 'test_dir') and os.path.exists(self.test_dir):
            try:
                import shutil
                shutil.rmtree(self.test_dir, ignore_errors=True)
            except:
                pass  # Ignore cleanup errors in Windows

class TestModelPerformanceAnalyzer(unittest.TestCase):
    """Test cases for performance analysis functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ModelPerformanceAnalyzer("LSTM")
        
        # Generate sample predictions
        np.random.seed(42)
        self.n_samples = 1000
        self.y_true = np.random.rand(self.n_samples) * 100000
        self.y_pred = self.y_true + np.random.normal(0, 5000, self.n_samples)
    
    def test_metrics_calculation(self):
        """Test metric calculations."""
        metrics = self.analyzer.calculate_metrics(
            self.y_true, self.y_pred
        )
        
        required_metrics = ['rmse', 'mae', 'r2', 'mape', 'direction_accuracy']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertTrue(np.isfinite(metrics[metric]))
            self.assertGreaterEqual(metrics[metric], 0)
    
    def test_visualization(self):
        """Test visualization functionality."""
        # Test actual vs predicted plot
        self.analyzer.plot_prediction_vs_actual(
            self.y_true, self.y_pred
        )
        plt.close()
        
        # Test residuals plot
        self.analyzer.plot_residuals(
            self.y_true, self.y_pred
        )
        plt.close()

class TestModelEvaluator(unittest.TestCase):
    """Test cases for model evaluation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = ModelEvaluator()
        self.metrics = PerformanceMetrics()
        self.viz = VisualizationTools()
        
        # Generate sample data
        np.random.seed(42)
        self.n_samples = 1000
        self.dates = pd.date_range('2020-01-01', periods=self.n_samples)
        self.y_true = np.random.rand(self.n_samples) * 100000
        self.y_pred = self.y_true + np.random.normal(0, 5000, self.n_samples)
    
    def test_regression_metrics(self):
        """Test regression metric calculations."""
        metrics = self.evaluator.calculate_regression_metrics(
            self.y_true, self.y_pred
        )
        
        required_metrics = ['mse', 'rmse', 'mae', 'mape', 'r2']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertTrue(np.isfinite(metrics[metric]))
    
    def test_prediction_intervals(self):
        """Test prediction interval calculation."""
        intervals = self.evaluator.calculate_prediction_intervals(
            self.y_pred, confidence_level=0.95
        )
        
        self.assertEqual(len(intervals), len(self.y_pred))
        self.assertTrue(all(low <= high for low, high in intervals))
        
        # Check coverage
        coverage = np.mean([low <= true <= high 
                          for (low, high), true 
                          in zip(intervals, self.y_true)])
        self.assertGreater(coverage, 0.9)  # Should be close to 0.95
    
    def test_custom_metrics(self):
        """Test custom performance metrics."""
        # Test MAPE
        mape = self.metrics.mean_absolute_percentage_error(
            self.y_true, self.y_pred
        )
        self.assertTrue(np.isfinite(mape))
        self.assertGreaterEqual(mape, 0)
        
        # Test price range accuracy
        accuracy = self.metrics.price_range_accuracy(
            self.y_true, self.y_pred, tolerance=0.1
        )
        self.assertTrue(0 <= accuracy <= 1)
        
        # Test weighted MAE
        weights = np.ones_like(self.y_true) / len(self.y_true)
        wmae = self.metrics.weighted_mean_absolute_error(
            self.y_true, self.y_pred, weights
        )
        self.assertTrue(np.isfinite(wmae))
        self.assertGreaterEqual(wmae, 0)
    
    def test_visualization(self):
        """Test visualization functionality."""
        # Test prediction plot
        fig = self.viz.plot_predictions(
            self.dates, self.y_true, self.y_pred
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
        
        # Test error distribution plot
        fig = self.viz.plot_error_distribution(
            self.y_true, self.y_pred
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
        
        # Test feature importance plot
        features = ['price', 'interest_rate', 'unemployment']
        importance = [0.5, 0.3, 0.2]
        fig = self.viz.plot_feature_importance(
            features, importance
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

if __name__ == '__main__':
    unittest.main()
