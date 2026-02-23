import pandas as pd
import numpy as np
import os
from datetime import datetime

def get_model_config(model):
    """Extract model architecture description from a Sequential model"""
    config = []
    for layer in model.layers:
        layer_name = layer.__class__.__name__
        
        if layer_name == 'Dense':
            units = layer.units
            activation = layer.activation.__name__ if layer.activation else 'linear'
            config.append(f"Dense({units}, {activation})")
        elif layer_name == 'Dropout':
            rate = layer.rate
            config.append(f"Dropout({rate})")
        elif layer_name == 'BatchNormalization':
            config.append("BatchNorm")
        else:
            config.append(layer_name)
    
    return " → ".join(config)

def count_parameters(model):
    """Count total trainable and non-trainable parameters"""
    trainable = sum([np.prod(w.shape) for w in model.trainable_weights])
    non_trainable = sum([np.prod(w.shape) for w in model.non_trainable_weights])
    return trainable + non_trainable

class TrialTracker:
    """Track and log model trials"""
    
    def __init__(self, csv_file='trials_results.csv', models_dir='saved_models'):
        self.csv_file = csv_file
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Load existing results or create new dataframe
        try:
            self.results_df = pd.read_csv(csv_file)
        except FileNotFoundError:
            self.results_df = pd.DataFrame()
    
    def log_trial(self, trial_name, model, metrics, config_dict=None, data_strategy=None):
        """Log a new trial with model and metrics"""
        
        model_config = get_model_config(model)
        total_params = count_parameters(model)
        trial_number = len(self.results_df) + 1
        
        trial_record = {
            'Trial #': trial_number,
            'Trial Name': trial_name,
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Data Strategy': data_strategy or 'None',
            'Model Architecture': model_config,
            'Total Parameters': total_params,
            'Test Accuracy': metrics.get('accuracy', 0),
            'Precision': metrics.get('precision', 0),
            'Recall': metrics.get('recall', 0),
            'F1-Score': metrics.get('f1', 0),
            'Train Loss': metrics.get('train_loss', 0),
            'Val Loss': metrics.get('val_loss', 0),
            'Epochs Trained': metrics.get('epochs', 0)
        }
        
        # Add optional config fields
        if config_dict:
            trial_record['Config'] = str(config_dict)
        
        # Append to dataframe
        self.results_df = pd.concat([self.results_df, pd.DataFrame([trial_record])], ignore_index=True)
        
        # Save to CSV
        self.results_df.to_csv(self.csv_file, index=False)
        print(f"\n✓ Trial #{trial_number} '{trial_name}' logged to {self.csv_file}")
        print(f"Total trials recorded: {len(self.results_df)}")
        
        return trial_number
    
    def get_trial_count(self):
        """Get the number of trials recorded so far"""
        return len(self.results_df)
    
    def save_model(self, model, trial_name, trial_number):
        """Save model with trial information"""
        model_filename = f"{self.models_dir}/{trial_name}_trial{trial_number}.h5"
        model.save(model_filename)
        print(f"✓ Model saved as '{model_filename}'")
        return model_filename
    
    def display_summary(self):
        """Display summary of all trials"""
        print("\n" + "="*100)
        print("TRIALS SUMMARY")
        print("="*100)
        print(self.results_df.to_string(index=False))
        print("="*100)