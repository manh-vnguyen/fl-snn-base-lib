from torch import nn
from .surrogates import TriangleSurr
from .utils import _initialize_weights, _supress_bn_bias


class LIFNode(nn.Module):
    def __init__(self, threshold, leak) -> None:
        super().__init__()
        self.spike_fn = TriangleSurr.apply
        self.threshold = threshold
        self.leak = leak
        self.mem = 0
    
    def reset(self):
        self.mem = 0
    
    def __call__(self, inp):
        self.mem = self.mem * self.leak + inp
        spk = self.spike_fn(self.mem)
        self.mem = self.mem - self.threshold * spk
        return spk

class BNLIFNode(LIFNode):
    def __init__(self, threshold, leak, bn, bn_target) -> None:
        super().__init__(threshold, leak)
        self.bn = bn
        self.bn_target = bn_target
    
    def __call__(self, inp):
        if self.bn_target == 'mem':
            self.mem = self.bn(self.mem * self.leak + inp)
        elif self.bn_target == 'spike':
            self.mem = self.mem * self.leak + self.bn(inp)
        spk = self.spike_fn(self.mem)
        self.mem = self.mem - self.threshold * spk
        return spk
    
class BNTTLIFNode(BNLIFNode):
    def __init__(self, threshold, leak, bn, bn_target) -> None:
        super().__init__(threshold, leak, bn, bn_target)
    
    def __call__(self, inp, t):
        if self.bn_target == 'mem':
            self.mem = self.bn[t](self.mem * self.leak + inp)
        elif self.bn_target == 'spike':
            self.mem = self.mem * self.leak + self.bn[t](inp)
        spk = self.spike_fn(self.mem)
        self.mem = self.mem - self.threshold * spk
        return spk

class SVGG(nn.Module):
    def __init__(self, 
                 conv_cfg,
                 fc_cfg,
                 threshold, timesteps, leak,
                 poisson_gen,
                 bn_type, bn_target,
                 bias_flag, bn_bias_flag,):
        super().__init__()
        assert bn_target in ['mem', 'spike', 'none'], f"bn_target must be 'mem', 'spike', or 'none', got {bn_target}"
        assert bn_type in ['bn', 'bntt', 'none'], f"bn_type must be 'bn', 'bntt', or 'none', got {bn_type}"
        assert (bn_target == 'none' and bn_type == 'none') or (bn_target in ['mem', 'spike'] and bn_type in ['bn', 'bntt']), f"Invalid combination: bn_target={bn_target}, bn_type={bn_type}. Either both must be 'none' or bn_target must be 'mem'/'spike' and bn_type must be 'bn'/'bntt'"
        self.timesteps = timesteps
        self.poisson_gen = poisson_gen
        
        # Initiate conv layers
        w = 32 # image size
        in_c = 3
        self.convs = []
        for x in conv_cfg:
            if not isinstance(x, str):
                self.convs.append(nn.Conv2d(in_c, x, kernel_size=3, stride=1, padding=1, bias=bias_flag))
                if bn_type == 'bntt':
                    self.convs.append(BNTTLIFNode(threshold, leak,
                            nn.ModuleList([nn.BatchNorm2d(x, eps=1e-5, momentum=0.1) for _ in range(self.timesteps)]),
                            bn_target))
                elif bn_type == 'bn':
                    self.convs.append(BNLIFNode(threshold, leak,
                            nn.BatchNorm2d(x, eps=1e-5, momentum=0.1),
                            bn_target))
                elif bn_type == 'none':
                    self.convs.append(LIFNode(threshold, leak))
                in_c = x
            elif x == 'A':
                w = w // 2
                self.convs.append(nn.AvgPool2d(kernel_size=2))
            elif x == 'M':
                w = w // 2
                self.convs.append(nn.AvgPool2d(kernel_size=2))
        
        self.convs = nn.ModuleList(self.convs)
        # Initiate fc layers
        self.fc_cfg = fc_cfg
        self.fcs = []
        prev = w*w*in_c
        for x in self.fc_cfg[:-1]:
            self.fcs.append(nn.Linear(prev, x, bias=bias_flag))
            if bn_type == 'bntt':
                self.fcs.append(BNTTLIFNode(threshold, leak,
                        nn.ModuleList([nn.BatchNorm1d(x, eps=1e-5, momentum=0.1) for _ in range(self.timesteps)]),
                        bn_target))
            elif bn_type == 'bn':
                self.fcs.append(BNLIFNode(threshold, leak,
                        nn.BatchNorm1d(x, eps=1e-5, momentum=0.1),
                        bn_target))
            prev = x
        self.fcs = nn.ModuleList(self.fcs)
        self.classifier = nn.Linear(self.fc_cfg[-2], self.fc_cfg[-1], bias=bias_flag)

        # Initialize the firing thresholds of all the layers
        _initialize_weights(self)
        if not bn_bias_flag:
            _supress_bn_bias(self)

    def reset(self):
        for m in self.modules():
            if isinstance(m, LIFNode):
                m.reset()

    def forward(self, inp):
        self.reset()
        batch_size = inp.size(0)
        mem_cls = 0
        for t in range(self.timesteps):
            spk = self.poisson_gen(inp)
            for m in self.convs:
                if isinstance(m, BNTTLIFNode):
                    spk = m(spk, t)
                else:
                    spk = m(spk)
            
            spk = spk.reshape(batch_size, -1)
            for m in self.fcs:
                if isinstance(m, BNTTLIFNode):
                    spk = m(spk, t)
                else:
                    spk = m(spk)

            mem_cls = mem_cls + self.classifier(spk)
        return mem_cls / self.timesteps