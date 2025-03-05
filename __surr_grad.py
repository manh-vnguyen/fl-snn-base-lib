import torch
import torch.nn.functional as F
import numpy as np

# 1. Sigmoid-based surrogate gradient
class SigmoidSurr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold=1.0, alpha=10.0):
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        ctx.threshold = threshold
        out = torch.zeros_like(input)
        out[input > threshold] = 1.0
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        alpha = ctx.alpha
        threshold = ctx.threshold
        # Sigmoid surrogate gradient
        grad_input = grad_output * alpha * torch.sigmoid(alpha * (input - threshold)) * (1 - torch.sigmoid(alpha * (input - threshold)))
        return grad_input, None, None

# 2. Arctan surrogate gradient
class ArctanSurr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold=1.0, alpha=2.0):
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        ctx.threshold = threshold
        out = torch.zeros_like(input)
        out[input > threshold] = 1.0
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        alpha = ctx.alpha
        threshold = ctx.threshold
        # Arctan surrogate gradient
        grad_input = grad_output * alpha / (1.0 + (np.pi/2 * alpha * (input - threshold))**2)
        return grad_input, None, None

# 3. Gaussian surrogate gradient
class GaussianSurr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold=1.0, sigma=0.5):
        ctx.save_for_backward(input)
        ctx.sigma = sigma
        ctx.threshold = threshold
        out = torch.zeros_like(input)
        out[input > threshold] = 1.0
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        sigma = ctx.sigma
        threshold = ctx.threshold
        # Gaussian surrogate gradient
        grad_input = grad_output * torch.exp(-(input - threshold)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
        return grad_input, None, None

# 4. Triangle surrogate gradient
class TriangleSurr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold=1.0, width=0.5):
        ctx.save_for_backward(input)
        ctx.width = width
        ctx.threshold = threshold
        out = torch.zeros_like(input)
        out[input > threshold] = 1.0
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        width = ctx.width
        threshold = ctx.threshold
        # Triangle (piecewise linear) surrogate gradient
        grad_input = torch.zeros_like(input)
        mask = (input >= threshold - width) & (input <= threshold + width)
        grad_input[mask] = 1.0 - torch.abs(input[mask] - threshold) / width
        return grad_output * grad_input, None, None

# 5. Exponential surrogate gradient
class ExponentialSurr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold=1.0, alpha=1.0):
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        ctx.threshold = threshold
        out = torch.zeros_like(input)
        out[input > threshold] = 1.0
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        alpha = ctx.alpha
        threshold = ctx.threshold
        # Exponential surrogate gradient
        grad_input = grad_output * alpha * torch.exp(-alpha * torch.abs(input - threshold))
        return grad_input, None, None

# 6. SuperSpike surrogate gradient (fast sigmoid)
class SuperSpikeSurr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold=1.0, beta=10.0):
        ctx.save_for_backward(input)
        ctx.beta = beta
        ctx.threshold = threshold
        out = torch.zeros_like(input)
        out[input > threshold] = 1.0
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        beta = ctx.beta
        threshold = ctx.threshold
        # SuperSpike surrogate gradient
        x = input - threshold
        grad_input = grad_output * beta / (1.0 + torch.abs(beta * x))**2
        return grad_input, None, None

# 7. Rectangular surrogate gradient
class RectangularSurr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold=1.0, width=0.5):
        ctx.save_for_backward(input)
        ctx.width = width
        ctx.threshold = threshold
        out = torch.zeros_like(input)
        out[input > threshold] = 1.0
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        width = ctx.width
        threshold = ctx.threshold
        # Rectangular surrogate gradient
        grad_input = torch.zeros_like(input)
        mask = (input >= threshold - width/2) & (input <= threshold + width/2)
        grad_input[mask] = 1.0 / width
        return grad_output * grad_input, None, None

# 8. Slayer surrogate gradient
class SlayerSurr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold=1.0, alpha=10.0):
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        ctx.threshold = threshold
        out = torch.zeros_like(input)
        out[input > threshold] = 1.0
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        alpha = ctx.alpha
        threshold = ctx.threshold
        # Slayer surrogate gradient (similar to SLAYER framework)
        x = input - threshold
        grad_input = grad_output * alpha * torch.exp(-alpha * torch.relu(x)) * torch.exp(-alpha * torch.relu(-x))
        return grad_input, None, None

# 9. Piecewise quadratic surrogate gradient
class QuadraticSurr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold=1.0, width=1.0):
        ctx.save_for_backward(input)
        ctx.width = width
        ctx.threshold = threshold
        out = torch.zeros_like(input)
        out[input > threshold] = 1.0
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        width = ctx.width
        threshold = ctx.threshold
        # Piecewise quadratic surrogate gradient
        grad_input = torch.zeros_like(input)
        mask = (input >= threshold - width) & (input <= threshold + width)
        x_norm = (input[mask] - threshold) / width
        grad_input[mask] = 0.75 * (1 - x_norm**2)
        return grad_output * grad_input, None, None

# 10. Soft LIF (Leaky Integrate-and-Fire) surrogate
class SoftLIFSurr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold=1.0, beta=1.0):
        ctx.save_for_backward(input)
        ctx.beta = beta
        ctx.threshold = threshold
        out = torch.zeros_like(input)
        out[input > threshold] = 1.0
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        beta = ctx.beta
        threshold = ctx.threshold
        # Soft LIF surrogate gradient
        x = input - threshold
        grad_input = grad_output * beta * torch.exp(beta * x) / (1 + torch.exp(beta * x))**2
        return grad_input, None, None

# 11. Piecewise linear surrogate
class PieceWiseLinearSurr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        ctx.save_for_backward(input)
        ctx.threshold = threshold # save the firing threshold for backward
        out = torch.zeros_like(input)
        out[input > 0] = threshold
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * 0.3 * F.threshold(ctx.threshold - torch.abs(input), 0, 0)
        return grad, None
    
# 12. Fast Sigmoid surrogate
class FastSigmoidSurr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold=1.0, alpha=10.0):
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        ctx.threshold = threshold
        out = torch.zeros_like(input)
        out[input > threshold] = 1.0
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        alpha = ctx.alpha
        threshold = ctx.threshold
        # Compute fast sigmoid surrogate gradient
        x = input - threshold
        grad_input = grad_output * alpha / (1.0 + torch.abs(alpha * x))**2
        return grad_input, None, None


if __name__ == '__main__':
    print(globals())