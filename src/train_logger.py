import csv
import os
from datetime import datetime
import yaml
import torch

class TrainLogger:
    def __init__(self, save_dir, log_file='log.csv'):
        """Initialize logger."""
        self.output_dir = save_dir
        self.log_path = os.path.join(self.output_dir, log_file)

        self.best_val_loss = float('inf')
        self.epoch = 0
        self.total_epochs = 0
        self.current_train_metrics = None
        self.smoothness_loss_sum = 0.0
        self.smoothness_loss_count = 0
        
        self.start_time = datetime.now()
        self.epoch_start_time = None
        
        self._init_log_files()
    
    def _init_log_files(self):
        """Initialize log file."""
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train_Loss', 'Train_Accuracy', 'Val_Loss', 'Val_Accuracy'])

    def _format_metrics(self, metrics):
        """Format metrics."""
        if 'train_loss' in metrics:
            return {
                'train_loss': f"{metrics['train_loss']:.4f}",
                'train_accuracy': f"{metrics['train_accuracy']:.2f}"
            }
        else:
            return {
                'val_loss': f"{metrics['val_loss']:.4f}",
                'val_accuracy': f"{metrics['val_accuracy']:.2f}"
            }

    def log_train_loss(self, metrics):
        """Log train metrics."""
        self.current_train_metrics = metrics

    def log_val_loss(self, metrics):
        """Log validation metrics."""
        if self.current_train_metrics:
            formatted_train = self._format_metrics(self.current_train_metrics)
            formatted_val = self._format_metrics(metrics)
            
            with open(self.log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.epoch,
                    formatted_train['train_loss'],
                    formatted_train['train_accuracy'],
                    formatted_val['val_loss'],
                    formatted_val['val_accuracy']
                ])
            
            self.current_train_metrics = None

    def start_epoch(self):
        """Record epoch start time."""
        self.epoch_start_time = datetime.now()
        self.smoothness_loss_sum = 0.0
        self.smoothness_loss_count = 0
    
    def update_smoothness_loss(self, loss_value):
        """Update smoothness loss."""
        self.smoothness_loss_sum += loss_value
        self.smoothness_loss_count += 1

    def get_time_info(self):
        """Get time info (epoch duration, remaining time)."""
        epoch_time = (datetime.now() - self.epoch_start_time).total_seconds()
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        remaining_time = (elapsed_time / (self.epoch + 1)) * (self.total_epochs - (self.epoch + 1))
        return epoch_time, remaining_time

    def print_log(self, epoch, train_metrics, val_metrics):
        """Print log."""
        formatted_train = self._format_metrics(train_metrics)
        formatted_val = self._format_metrics(val_metrics)
        
        epoch_time, remaining_time = self.get_time_info()
        
        print(f'Time per epoch: {epoch_time:.2f}s | '
              f'Estimated time to finish: {remaining_time/60:.2f}min')
        print(f'Train Loss: {formatted_train["train_loss"]} | '
              f'Accuracy: {formatted_train["train_accuracy"]}%')
        
        # Smoothness loss
        if self.smoothness_loss_count > 0:
            avg_smoothness_loss = self.smoothness_loss_sum / self.smoothness_loss_count
            print(f'CSS Smoothness Loss: {avg_smoothness_loss:.6f}')
        print(f'Val Loss: {formatted_val["val_loss"]} | '
              f'Accuracy: {formatted_val["val_accuracy"]}%')

    def update_batch_progress(self, current_batch, total_batches, phase):
        """Update batch progress."""
        progress = current_batch / total_batches * 100
        print(f'\r{phase} Progress: {progress:.1f}% [{current_batch}/{total_batches}]', end='')
        if current_batch == total_batches:
            print()

    def update_epoch(self, current_epoch, total_epochs):
        """Update epoch."""
        self.epoch = current_epoch
        self.total_epochs = total_epochs
        print(f'\nEpoch [{current_epoch}/{total_epochs}]')

    def save_checkpoint(self, epoch, css_model, gamma_model, ccm_model, classification_model, is_best=False):
        """Save checkpoint. best_model/: overwrite when val_loss improves. latest/: overwrite every epoch."""
        if is_best:
            models_dir = os.path.join(self.output_dir, 'best_model')
        else:
            models_dir = os.path.join(self.output_dir, 'latest')
        os.makedirs(models_dir, exist_ok=True)

        # Save each model separately
        css_path = os.path.join(models_dir, 'css_model.pth')
        torch.save(css_model.state_dict(), css_path)

        gamma_path = os.path.join(models_dir, 'gamma_model.pth')
        torch.save(gamma_model.state_dict(), gamma_path)

        ccm_path = os.path.join(models_dir, 'ccm_model.pth')
        torch.save(ccm_model.state_dict(), ccm_path)

        classification_path = os.path.join(models_dir, 'classification_model.pth')
        torch.save(classification_model.state_dict(), classification_path)

        model_info = {
            'epoch': epoch,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'is_best': is_best
        }
        info_path = os.path.join(models_dir, 'model_info.yaml')
        with open(info_path, 'w') as f:
            yaml.dump(model_info, f)

        label = 'best_model' if is_best else 'latest'
        print(f'Checkpoint saved: {models_dir}')

    def log_message(self, message):
        """Log a message."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}\n"
        
        print(log_entry.strip())
        message_log_path = os.path.join(self.output_dir, 'messages.log')
        with open(message_log_path, 'a') as f:
            f.write(log_entry)