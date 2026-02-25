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

class Gamma(nn.Module):
    def __init__(self, init_gamma = 1 / 2.2, trainable=True):
        super(Gamma, self).__init__()
        self.trainable = trainable
        self.gamma = nn.Parameter(torch.tensor(init_gamma))
        self.gamma.requires_grad = trainable

    def forward(self, x):
        return x.pow(self.gamma)

class SimplifiedGradientLimitedGammaFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma_param, grad_th_tensor):
        # gamma_param の勾配計算には影響させずに th と a を計算するため .detach() を使用
        gamma_detached = gamma_param.detach()

        # 元の GradientLimitedGammaV1 のフォワード計算ロジック
        # (gamma_detached - 1.0) がほぼ0の場合、exponent_for_th が非常に大きくなる可能性があるが、
        # torch.pow がこれを処理することを期待する。
        # 元のコードもこの部分に特別な安定化策はなかったため、挙動を維持。
        th_base = grad_th_tensor / gamma_detached
        exponent_for_th = 1.0 / (gamma_detached - 1.0) 
        
        th = th_base.pow(exponent_for_th)
        
        # a = th.pow(gamma_detached) / th と同等
        a_coeff = th.pow(gamma_detached - 1.0)

        m = (x > th).to(x.dtype)
        
        # x.pow(gamma_param) の計算には学習対象の gamma_param を使用
        output = (1 - m) * a_coeff * x + m * x.pow(gamma_param)

        ctx.save_for_backward(x, gamma_param, m, a_coeff, th)
        # grad_th_tensor は定数として扱われるため、勾配計算は不要
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, gamma_param, m, a_coeff, th = ctx.saved_tensors
        
        grad_x = None
        grad_gamma = None

        # x に対する勾配計算
        # d_output_dx = (1 - m) * a_coeff + m * gamma_param * x.pow(gamma_param - 1)
        # m=1 の場合 (x > th)、th > 0 であれば x > 0 となり、x.pow(gamma_param - 1) は安全。
        # (th <= 0 となるケースは通常想定されない)
        
        deriv_pow_term_wrt_x_values = torch.zeros_like(x)
        active_mask_m1 = m > 0.5 # m == 1.0 となる箇所

        # active_mask_m1 が全て False の場合でも、以下の処理は PyTorch によって適切に処理されます。
        # x_active は空テンソルになり、pow_term_derivative_for_active も空テンソルになります。
        # deriv_pow_term_wrt_x_values への代入も空のスライスへの代入となり、結果的にゼロのままです。
        x_active = x[active_mask_m1] # x > th > 0 を想定
        
        # アクティブな要素に対してのみ微分項を計算
        pow_term_derivative_for_active = gamma_param * x_active.pow(gamma_param - 1.0)
        
        # 計算結果を元のテンソルの対応する位置に代入
        deriv_pow_term_wrt_x_values[active_mask_m1] = pow_term_derivative_for_active
        
        grad_x_contribution = (1 - m) * a_coeff + m * deriv_pow_term_wrt_x_values
        grad_x = grad_output * grad_x_contribution

        # gamma_param に対する勾配計算
        # d_output_dgamma = m * x.pow(gamma_param) * log(x)
        # ClippedPower の grad_a の計算方法を参考に log(0) を対処
        
        # gamma_param がスカラー nn.Parameter であると仮定
        is_gamma_positive = gamma_param.item() > 0 if gamma_param.numel() == 1 else (gamma_param > 0)
        
        dpow_dgamma_elementwise = torch.where(
            (x == 0) & is_gamma_positive, # x=0 かつ gamma > 0 ならば、勾配は0
            torch.zeros_like(x),
            # それ以外の場合 (x > 0, または x=0 で gamma <=0 の problematic なケースも含む)
            x.pow(gamma_param) * torch.log(x.clamp(min=1e-38)) # x>0 付近の log(0) と、gamma<=0 時の x=0 を clamp でケア
        )
        
        # gamma_param はスカラーなので、勾配は合計する
        grad_gamma = (grad_output * m * dpow_dgamma_elementwise).sum()
        
        return grad_x, grad_gamma, None # grad_th_tensor に対する勾配は None

class GradientLimitedGammaV1(nn.Module):
    def __init__(self, init_gamma = 1 / 2.2, grad_th = 2.0, trainable=True):
        super(GradientLimitedGammaV1, self).__init__()
        self.gamma = nn.Parameter(torch.tensor(init_gamma, dtype=torch.float32))
        self.gamma.requires_grad = trainable
        # grad_th をテンソルとして保持 (バッファとして登録も可能)
        self.grad_th = torch.tensor(grad_th, dtype=torch.float32)

    def forward(self, x):
        # grad_th を x と同じデバイスに配置して .apply に渡す
        return SimplifiedGradientLimitedGammaFunc.apply(x, self.gamma, self.grad_th.to(x.device))

class GradientLimitedGammaV2(nn.Module):
    def __init__(self, init_gamma=1/2.2, grad_th=2.0, trainable=True):
        super(GradientLimitedGammaV2, self).__init__()
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

        # Avoid NaN from x.log() when x is 0 and a is positive.
        # The gradient of x^a * log(x) w.r.t. a is 0 if x is 0 and a > 0.
        # If x is 0 and a <= 0, it's undefined, but x^a would be inf or nan,
        # which should be handled by the forward pass or other parts of the model.
        # Here we focus on the case where x=0 leads to 0 * -inf for grad_a.
        grad_a = torch.where(
            (x == 0) & (a > 0), 
            torch.zeros_like(x), 
            grad_output * (x ** a) * torch.log(x.clamp(min=1e-38)) # clamp to avoid log(0)
        )
        grad_a_clipped = torch.clamp(grad_a, min=-th, max=th)

        return grad_x_clipped, grad_a_clipped, None

# 1. 微小値加算
class ClippedPowerEpsAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, eps=1e-7): # epsはデフォルト値として保持
        x_added = x + eps
        ctx.save_for_backward(x_added, a)
        # forwardではepsを返さないので、backwardの入力数と合わせるためにNoneを追加
        return x_added ** a

    @staticmethod
    def backward(ctx, grad_output):
        x, a = ctx.saved_tensors
        # clamp to avoid log(0) or log(negative) from eps addition if x was very small negative
        # x in backward is x_clipped from forward, which is already >= eps
        grad_x = grad_output * a * x ** (a - 1)
        grad_a = grad_output * x ** a * torch.log(x) # x is guaranteed to be > 0
        return grad_x, grad_a, None # grad_eps is None

class GammaEpsAdd(nn.Module):
    def __init__(self, init_gamma=1/2.2, trainable=True, eps=1e-7, grad_th=None): # grad_th is unused but kept for compatibility
        super(GammaEpsAdd, self).__init__()
        self.trainable = trainable
        self.gamma = nn.Parameter(torch.tensor(init_gamma))
        self.gamma.requires_grad = trainable
        self.eps = eps

    def forward(self, x):
        return ClippedPowerEpsAdd.apply(x, self.gamma, self.eps)

# 2. εクリッピング
class ClippedPowerEpsClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, eps=1e-7): # epsはデフォルト値として保持
        x_clipped = torch.clamp(x, min=eps)
        ctx.save_for_backward(x_clipped, a)
        return x_clipped ** a

    @staticmethod
    def backward(ctx, grad_output):
        x, a = ctx.saved_tensors
        # x in backward is x_clipped from forward, which is already >= eps
        grad_x = grad_output * a * x ** (a - 1)
        grad_a = grad_output * x ** a * torch.log(x) # x is guaranteed to be > 0
        return grad_x, grad_a, None # grad_eps is None

class GammaEpsClip(nn.Module):
    def __init__(self, init_gamma=1/2.2, trainable=True, eps=1e-7, grad_th=None): # grad_th is unused but kept for compatibility
        super(GammaEpsClip, self).__init__()
        self.trainable = trainable
        self.gamma = nn.Parameter(torch.tensor(init_gamma))
        self.gamma.requires_grad = trainable
        self.eps = eps

    def forward(self, x):
        return ClippedPowerEpsClip.apply(x, self.gamma, self.eps)
