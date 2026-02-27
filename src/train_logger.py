import csv
import os
from datetime import datetime
import yaml
import torch

class TrainLogger:
    def __init__(self, save_dir, log_file='log.csv'):
        """ロガーの初期化"""
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
        """ログファイルの初期化"""
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train_Loss', 'Train_Accuracy', 'Val_Loss', 'Val_Accuracy'])

    def _format_metrics(self, metrics):
        """メトリクスをフォーマットする"""
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
        """訓練メトリクスの記録"""
        self.current_train_metrics = metrics

    def log_val_loss(self, metrics):
        """検証メトリクスの記録"""
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
        """エポックの開始時間を記録"""
        self.epoch_start_time = datetime.now()
        self.smoothness_loss_sum = 0.0
        self.smoothness_loss_count = 0
    
    def update_smoothness_loss(self, loss_value):
        """smoothness lossを更新"""
        self.smoothness_loss_sum += loss_value
        self.smoothness_loss_count += 1

    def get_time_info(self):
        """時間情報を取得する
        
        Returns:
            tuple: (エポック所要時間, 残り時間)
        """
        epoch_time = (datetime.now() - self.epoch_start_time).total_seconds()
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        remaining_time = (elapsed_time / (self.epoch + 1)) * (self.total_epochs - (self.epoch + 1))
        return epoch_time, remaining_time

    def print_log(self, epoch, train_metrics, val_metrics):
        """ログを表示する"""
        formatted_train = self._format_metrics(train_metrics)
        formatted_val = self._format_metrics(val_metrics)
        
        epoch_time, remaining_time = self.get_time_info()
        
        print(f'Time per epoch: {epoch_time:.2f}s | '
              f'Estimated time to finish: {remaining_time/60:.2f}min')
        print(f'Train Loss: {formatted_train["train_loss"]} | '
              f'Accuracy: {formatted_train["train_accuracy"]}%')
        
        # Smoothness lossの表示
        if self.smoothness_loss_count > 0:
            avg_smoothness_loss = self.smoothness_loss_sum / self.smoothness_loss_count
            print(f'CSS Smoothness Loss: {avg_smoothness_loss:.6f}')
        print(f'Val Loss: {formatted_val["val_loss"]} | '
              f'Accuracy: {formatted_val["val_accuracy"]}%')

    def update_batch_progress(self, current_batch, total_batches, phase):
        """バッチの進捗を更新する"""
        progress = current_batch / total_batches * 100
        print(f'\r{phase} Progress: {progress:.1f}% [{current_batch}/{total_batches}]', end='')
        if current_batch == total_batches:
            print()

    def update_epoch(self, current_epoch, total_epochs):
        """エポックを更新する"""
        self.epoch = current_epoch
        self.total_epochs = total_epochs
        print(f'\nEpoch [{current_epoch}/{total_epochs}]')

    def save_checkpoint(self, epoch, css_model, gamma_model, ccm_model, classification_model, is_best=False):
        """チェックポイントを保存する
        
        best_model/: val_loss が最良のときのみ上書き保存
        latest/: 毎エポック上書き保存（常に best と latest の2つ分のみ保持）
        
        Args:
            epoch: 現在のエポック
            css_model: CSSモデル
            gamma_model: ガンマモデル
            ccm_model: CCMモデル
            classification_model: 分類モデル
            is_best: True のとき best_model/, False のとき latest/ に保存
        """
        if is_best:
            models_dir = os.path.join(self.output_dir, 'best_model')
        else:
            models_dir = os.path.join(self.output_dir, 'latest')
        os.makedirs(models_dir, exist_ok=True)

        # 各モデルを個別のファイルに保存（パラメータごと）
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
        print(f'モデルを保存しました: {models_dir}')

    def log_message(self, message):
        """任意のメッセージをログに記録する"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}\n"
        
        # コンソールに出力
        print(log_entry.strip()) # printは改行を自動で入れるのでstrip()
        
        # ログファイルに追記 (オプショナル)
        # 必要であれば、別のログファイルを用意するか、既存のcsvとは別にテキストファイルなどを用意
        # ここでは、専用のメッセージログファイル 'messages.log' を作成する例
        message_log_path = os.path.join(self.output_dir, 'messages.log')
        with open(message_log_path, 'a') as f:
            f.write(log_entry)