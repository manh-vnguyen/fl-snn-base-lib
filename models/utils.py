import torch
from torch import nn
import math

def _initialize_weights(net):
    generator = torch.Generator().manual_seed(42)
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5), generator=generator)
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                if fan_in != 0:
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound, generator=generator)

def _supress_bn_bias(net):
    for m in net.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.bias = None

class PoissonGen():
    def __init__(self, rescale_fac=1.0, neg_spike=True):
        self.rescale_fac = rescale_fac
        self.neg_spike = neg_spike
    
    def __call__(self, inp):
        if self.neg_spike:
            return torch.mul(torch.rand_like(inp * self.rescale_fac).le(torch.abs(inp)).float(), torch.sign(inp))
        else:
            return torch.rand_like(inp * self.rescale_fac).le(inp).to(inp)