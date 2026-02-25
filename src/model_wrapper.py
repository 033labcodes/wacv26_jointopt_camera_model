import torch

class ModelWrapper:
    def __init__(self, css_model, gamma_model, ccm_model, classification_model, criterion, optimizer, lr_schedule, device, logger, camera_name, css_smoothness_weight=0.0):
        """モデルラッパーの初期化"""
        self.css_model = css_model
        self.gamma_model = gamma_model
        self.ccm_model = ccm_model
        self.classification_model = classification_model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.device = device
        self.logger = logger
        self.camera_name = camera_name
        self.css_smoothness_weight = css_smoothness_weight

    def process_batch(self, inputs_hsi, target, training=True):
        """バッチの処理"""
        inputs_hsi = inputs_hsi.to(self.device)
        target = target.to(self.device)
        
        rgb_images = self.css_model(inputs_hsi)
        rgb_images = self.ccm_model(rgb_images)
        rgb_images = rgb_images.clamp(min=0)
        rgb_images = self.gamma_model(rgb_images)

        if self.camera_name == 'gray':
            gray_values = rgb_images.mean(dim=1, keepdim=True)
            rgb_images = gray_values.expand(-1, 3, -1, -1)
        
        if training:
            self.classification_model.train()
            outputs = self.classification_model(rgb_images)
            classification_loss = self.criterion(outputs, target)
            
            # CSS smoothness lossを追加
            total_loss = classification_loss
            if self.css_smoothness_weight > 0 and self.css_model.trainable:
                smoothness_loss = self.css_model.compute_smoothness_loss()
                total_loss = classification_loss + self.css_smoothness_weight * smoothness_loss
                self.logger.update_smoothness_loss(smoothness_loss.item())
            
            total_loss.backward()
            self.optimizer.step()
            if self.camera_name != 'sRGB':
                self.css_model.normalize_weights()
            
            loss = classification_loss  # メトリクス用に分類損失を保持
        else:
            self.classification_model.eval()
            with torch.no_grad():
                outputs = self.classification_model(rgb_images)
                loss = self.criterion(outputs, target)
        
        return loss, outputs, rgb_images
    
    def train(self, train_loader):
        """モデルを訓練する"""
        self.css_model.train()
        self.gamma_model.train()
        self.ccm_model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        num_batches = len(train_loader)
        
        for i, (inputs_hsi, target) in enumerate(train_loader):
            self.logger.update_batch_progress(i+1, num_batches, 'Training')
            self.optimizer.zero_grad()
            
            loss, outputs, _ = self.process_batch(inputs_hsi, target, training=True)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target.to(self.device)).sum().item()
        
        if isinstance(self.lr_schedule, torch.optim.lr_scheduler.StepLR):
            self.lr_schedule.step()
        
        return {'train_loss': total_loss / num_batches, 'train_accuracy': 100. * correct / total}
    
    def evaluate(self, val_loader):
        """モデルを評価する"""
        self.css_model.eval()
        self.gamma_model.eval()
        self.ccm_model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        num_batches = len(val_loader)

        with torch.no_grad():
            for i, (inputs_hsi, target) in enumerate(val_loader):
                self.logger.update_batch_progress(i+1, num_batches, 'Validation')
                loss, outputs, rgb_images = self.process_batch(inputs_hsi, target, training=False)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target.to(self.device)).sum().item()            
        return {'val_loss': total_loss / num_batches, 'val_accuracy': 100. * correct / total}
