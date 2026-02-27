import torch
import torch.nn as nn
import torch.nn.functional as F

class CSSModel(nn.Module):
    def __init__(self, init_weights: torch.Tensor = None, in_channels=20, out_channels=3, trainable=True):
        super().__init__()
        self.trainable = trainable

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        if init_weights is not None:
            self.conv.weight.data = init_weights

        if not trainable:
            self.conv.weight.requires_grad = False

    def forward(self, x):
        x = self.conv(x)
        return x
        
    def normalize_weights(self):
        weights = self.conv.weight.data.clamp(min=0, max=1)
        self.conv.weight.data = weights
    
    def compute_smoothness_loss(self):
        """
        CSS感度関数の滑らかさ制約を計算
        隣接する波長間の差の二乗平均を返す
        """
        weights = self.conv.weight.squeeze(-1).squeeze(-1)
        
        diff = weights[:, 1:] - weights[:, :-1]
        
        smoothness_loss = torch.mean(diff ** 2)
        
        return smoothness_loss