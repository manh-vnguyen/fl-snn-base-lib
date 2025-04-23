import torch
from torch import nn
import __surr_grad as surr_lib

class ANN_VGG(nn.Module):
    def __init__(self, num_cls=10, kernel_size=3, dropout=0.2):
        super(ANN_VGG, self).__init__()
        
        self.kernel_size    = kernel_size
        self.dropout        = dropout
        self.features       = self._make_layers()

        self.classifier = nn.Sequential(
            nn.Linear(256*2*2, 1024, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            # nn.Linear(1024, 1024, bias=False),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(1024, num_cls, bias=False)
        )
        

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    
    def _make_layers(self, cfg=[64, 'A', 64, 128, 'A', 128, 256, 'A', 256, 'A', 256]):
        layers = []
        in_channels = 3
        for x in cfg:
            stride = 1
            
            if x == 'A':
                layers.pop()
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2, stride=stride, bias=False),
                        nn.BatchNorm2d(x),
                        nn.ReLU(inplace=True)
                        ]
                layers += [nn.Dropout(self.dropout)]
                in_channels = x        
        return nn.Sequential(*layers)

def PoissonGen(inp, rescale_fac=2.0):
    rand_inp = torch.rand_like(inp)
    return torch.mul(torch.le(rand_inp * rescale_fac, torch.abs(inp)).float(), torch.sign(inp))

class SNN_VGG(nn.Module):
    def __init__(self, img_size=32,  num_cls=10, 
                 surrogate={'type': 'TriangleSurr', 'params': {}}, 
                 threshold=1.0, timesteps=25, leak=0.95, 
                 device=torch.device('cuda:0')):
        super(SNN_VGG, self).__init__()
        self.device = device
        self.img_size = img_size
        self.num_cls = num_cls
        self.timesteps = timesteps
        self.spike_fn = getattr(surr_lib, surrogate['type']).apply
        self.surr_params = surrogate['params']
        self.leak = leak
        self.batch_num = self.timesteps
        self.threshold = threshold

        affine_flag = True
        bias_flag = False

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt1 = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt2 = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool1 = nn.AvgPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt3 = nn.ModuleList([nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt4 = nn.ModuleList([nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool2 = nn.AvgPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt5 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt6 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt7 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool3 = nn.AvgPool2d(kernel_size=2)

        self.fc1 = nn.Linear((self.img_size//8)*(self.img_size//8)*256, 1024, bias=bias_flag)
        self.bntt_fc = nn.ModuleList([nn.BatchNorm1d(1024, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.fc2 = nn.Linear(1024, self.num_cls, bias=bias_flag)

        self.conv_list = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7]
        self.bntt_list = [self.bntt1, self.bntt2, self.bntt3, self.bntt4, self.bntt5, self.bntt6, self.bntt7, self.bntt_fc]
        self.pool_list = [False, self.pool1, False, self.pool2, False, False, self.pool3]

        # Turn off bias of BNTT
        for bn_list in self.bntt_list:
            for bn_temp in bn_list:
                bn_temp.bias = None


        # Initialize the firing thresholds of all the layers
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                torch.nn.init.xavier_uniform_(m.weight, gain=2)



    def forward(self, inp):
        batch_size = inp.size(0)
        mem_conv1 = torch.zeros(batch_size, 64, self.img_size, self.img_size).to(self.device)
        mem_conv2 = torch.zeros(batch_size, 64, self.img_size, self.img_size).to(self.device)
        mem_conv3 = torch.zeros(batch_size, 128, self.img_size//2, self.img_size//2).to(self.device)
        mem_conv4 = torch.zeros(batch_size, 128, self.img_size//2, self.img_size//2).to(self.device)
        mem_conv5 = torch.zeros(batch_size, 256, self.img_size//4, self.img_size//4).to(self.device)
        mem_conv6 = torch.zeros(batch_size, 256, self.img_size//4, self.img_size//4).to(self.device)
        mem_conv7 = torch.zeros(batch_size, 256, self.img_size//4, self.img_size//4).to(self.device)
        mem_conv_list = [mem_conv1, mem_conv2, mem_conv3, mem_conv4, mem_conv5, mem_conv6, mem_conv7]

        mem_fc1 = torch.zeros(batch_size, 1024).to(self.device)
        mem_fc2 = torch.zeros(batch_size, self.num_cls).to(self.device)

        for t in range(self.timesteps):
            spike_inp = PoissonGen(inp)
            out_prev = spike_inp

            for i in range(len(self.conv_list)):
                mem_conv_list[i] = self.leak * mem_conv_list[i] + self.bntt_list[i][t](self.conv_list[i](out_prev))
                out = self.spike_fn(mem_conv_list[i], self.threshold, **self.surr_params)
                mem_conv_list[i] = mem_conv_list[i] - out * self.threshold
                out_prev = out.clone()

                if self.pool_list[i] is not False:
                    out = self.pool_list[i](out_prev)
                    out_prev = out.clone()

            out_prev = out_prev.reshape(batch_size, -1)

            mem_fc1 = self.leak * mem_fc1 + self.bntt_fc[t](self.fc1(out_prev))
            out = self.spike_fn(mem_fc1, self.threshold, **self.surr_params)
            mem_fc1 = mem_fc1 - out * self.threshold
            out_prev = out.clone()

            mem_fc2 = mem_fc2 + self.fc2(out_prev)

        out_voltage = mem_fc2 / self.timesteps

        return out_voltage

class ANN_FC(nn.Module):
    def __init__(self):
        super(ANN_FC, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 100)
        self.fc2 = nn.Linear(100, 10)
    def forward(self, x):
        x = torch.relu(x.view(x.size(0), -1))
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SNN_FC(nn.Module):
    def __init__(self, surrogate={'type': 'TriangleSurr', 'params': {}}, threshold=1.0, timesteps=8, leak=0.95, device=torch.device('cuda:5')):
        super(SNN_FC, self).__init__()
        self.device = device
        self.num_cls = 10
        self.timesteps = timesteps
        self.spike_fn = getattr(surr_lib, surrogate['type']).apply
        self.surr_params = surrogate['params']
        self.leak = leak
        self.threshold = threshold

        bias_flag = True

        self.fc1 = nn.Linear(28*28, 100, bias=bias_flag)
        self.fc2 = nn.Linear(100, self.num_cls, bias=bias_flag)

        # Initialize the firing thresholds of all the layers
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                torch.nn.init.xavier_uniform_(m.weight, gain=2)



    def forward(self, inp):
        batch_size = inp.size(0)
        mem_fc1 = torch.zeros(batch_size, 100).to(self.device)
        mem_fc2 = torch.zeros(batch_size, self.num_cls).to(self.device)

        for t in range(self.timesteps):
            spike = PoissonGen(inp).reshape(batch_size, -1)

            mem_fc1 = self.leak * mem_fc1 + self.fc1(spike)
            out = self.spike_fn(mem_fc1, self.threshold, **self.surr_params)
            mem_fc1 = mem_fc1 - out * self.threshold
            
            spike = out.clone()
            mem_fc2 = mem_fc2 + self.fc2(spike)

        out_voltage = mem_fc2 / self.timesteps

        return out_voltage


def init_model(args, device, num_cls):
    if args.model == 'snn_vgg9':
        return SNN_VGG(img_size=32, 
                    num_cls=num_cls,
                    device=device, 
                    **args.snn_hyperparams).to(device)
    
    if args.model == 'ann_vgg9':
        return ANN_VGG(num_cls=num_cls).to(device)

    if args.model == 'ann_fc':
        return ANN_FC().to(device)
    
    if args.model == 'snn_fc':
        return SNN_FC(device=device,
                      **args.snn_hyperparams).to(device)


if __name__ == '__main__':
    model = SNN_VGG(device='cpu')