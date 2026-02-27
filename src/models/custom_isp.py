import torch
import torch.nn as nn
import torch.nn.functional as F
    

class ColorCorrectionMatrix(nn.Module):
    def __init__(self, init_ccm = torch.eye(3), trainable=True, scaling_factor=1.0):
        super().__init__()
        self.trainable = trainable
        self.scaling_factor = scaling_factor
        
        self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv.weight.data = init_ccm.view(3, 3, 1, 1) * self.scaling_factor
        
    def forward(self, x):
        x = self.conv(x)
        return x

class DerivativeClippingGamma(nn.Module):
    def __init__(self, init_gamma=1/2.2, grad_th=2.0, trainable=True):
        super(DerivativeClippingGamma, self).__init__()
        self.trainable = trainable
        self.gamma = nn.Parameter(torch.tensor(init_gamma))
        self.grad_th = grad_th
        self.gamma.requires_grad = trainable

    def forward(self, x):
        return ClippedPower.apply(x, self.gamma, self.grad_th)

class ClippedPower(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, th):
        ctx.save_for_backward(x, a)
        ctx.th = th
        return x ** a

    @staticmethod
    def backward(ctx, grad_output):
        x, a = ctx.saved_tensors
        th = ctx.th

        grad_x = grad_output * a * x ** (a - 1)
        grad_x_clipped = torch.clamp(grad_x, min=-th, max=th)

        grad_a = torch.where(
            (x == 0) & (a > 0), 
            torch.zeros_like(x), 
            grad_output * (x ** a) * torch.log(x.clamp(min=1e-38)) # clamp to avoid log(0)
        )
        grad_a_clipped = torch.clamp(grad_a, min=-th, max=th)

        return grad_x_clipped, grad_a_clipped, None