import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json

class TrainingMonitor:
    _instance = None
    
    def __new__(cls, model_name):
        if cls._instance is None:
            cls._instance = super(TrainingMonitor, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, model_name):
        if not hasattr(self, 'initialized'):
            self.model_name = model_name
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_dir = os.path.join(os.getcwd(), "training_logs", f"{self.model_name}_{self.timestamp}")
            os.makedirs(self.save_dir, exist_ok=True)
            
            self.fig, self.ax = plt.subplots(figsize=(10, 6))
            self.fig.canvas.manager.set_window_title(f'Training Monitor - {model_name}')
            self.initialized = True
            self.loss_history = []
    
    def update(self, mse_loss, phys_loss, lr):
        self.loss_history.append({
            'epoch': len(self.loss_history)+1,
            'mse': mse_loss,
            'physics': phys_loss,
            'lr': lr
        })
        self._plot()
        self._save()
    
    def _plot(self):
        self.ax.clear()
        epochs = [x['epoch'] for x in self.loss_history]
        mse = [x['mse'] for x in self.loss_history]
        phys = [x['physics'] for x in self.loss_history]
        
        self.ax.plot(epochs, mse, 'b-', label='MSE Loss', linewidth=2)
        self.ax.plot(epochs, phys, 'r-', label='Physics Loss', linewidth=2)
        self.ax.set_xlabel('Epoch', fontsize=12)
        self.ax.set_ylabel('Loss', fontsize=12)
        self.ax.set_title(f'Training Progress - {self.model_name}', fontsize=14)
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.legend(fontsize=10)
        
        plt.tight_layout()
        self.fig.canvas.draw()
    
    def _save(self):
        log_path = os.path.join(self.save_dir, 'training_log.json')
        with open(log_path, 'w') as f:
            json.dump({
                'model_name': self.model_name,
                'timestamp': self.timestamp,
                'history': self.loss_history
            }, f, indent=2)
        
        plot_path = os.path.join(self.save_dir, 'loss_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    def show(self):
        plt.show(block=True)
