import torch

class TriangleSurr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold=1.0, alpha = 1.0, dampen=0.3):
        ctx.save_for_backward(input)
        ctx.threshold = threshold
        ctx.alpha = alpha
        ctx.dampen = dampen
        return (input > threshold).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        threshold = ctx.threshold
        alpha = ctx.alpha
        dampen = ctx.dampen
        x = input - threshold
        grad_input = grad_output * dampen * torch.maximum(threshold - torch.abs(alpha * x), torch.tensor(0))
        return grad_input, None, None

class FastSigmoidSurr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold=1.0, alpha=1.5, dampen=0.3):
        ctx.save_for_backward(input)
        ctx.threshold = threshold
        ctx.alpha = alpha
        ctx.dampen = dampen
        return (input > threshold).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        threshold = ctx.threshold
        alpha = ctx.alpha
        dampen = ctx.dampen
        x = input - threshold
        grad_input = grad_output * dampen / (1.0 + torch.abs(alpha * x))**2
        return grad_input, None, None
    
class GaussianSurr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold=1.0, sigma=0.4, dampen=0.3):
        ctx.save_for_backward(input)
        ctx.threshold = threshold
        ctx.sigma = sigma
        ctx.dampen = dampen
        return (input > threshold).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        threshold = ctx.threshold
        sigma = ctx.sigma
        dampen = ctx.dampen
        x = input - threshold
        grad_input = grad_output * dampen * torch.exp(-x**2/(2*sigma**2)) / (sigma * torch.sqrt(torch.tensor(2 * torch.pi)))
        return grad_input, None, None
    
class QuadraticSurr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold=1.0, width=1.0, dampen=0.3):
        ctx.save_for_backward(input)
        ctx.threshold = threshold
        ctx.width = width
        ctx.dampen = dampen
        return (input > threshold).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        threshold = ctx.threshold
        width = ctx.width
        dampen = ctx.dampen
        x = input - threshold
        grad_input = torch.zeros_like(input)
        mask = torch.abs(x) <= width
        grad_input[mask] = dampen * (1 - (x[mask]/width)**2)
        return grad_output * grad_input, None, None
    
class RectangleSurr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold=1.0, width=1.0, dampen=0.3):
        ctx.save_for_backward(input)
        ctx.threshold = threshold
        ctx.width = width
        ctx.dampen = dampen
        return (input > threshold).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        threshold = ctx.threshold
        width = ctx.width
        dampen = ctx.dampen
        x = input - threshold
        grad_input = torch.zeros_like(input)
        mask = torch.abs(x) <= width/2
        grad_input[mask] =  dampen / width
        return grad_output * grad_input, None, None