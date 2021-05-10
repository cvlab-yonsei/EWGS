import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

__all__ = ['QConv']

class EWGS_discretizer(torch.autograd.Function):
    """
    x_in: continuous inputs within the range of [0,1]
    num_levels: number of discrete levels
    scaling_factor: backward scaling factor
    x_out: discretized version of x_in within the range of [0,1]
    """
    @staticmethod
    def forward(ctx, x_in, num_levels, scaling_factor):
        x = x_in * (num_levels - 1)
        x = torch.round(x)
        x_out = x / (num_levels - 1)
        
        ctx._scaling_factor = scaling_factor
        ctx.save_for_backward(x_in-x_out)
        return x_out
    @staticmethod
    def backward(ctx, g):
        diff = ctx.saved_tensors[0]
        delta = ctx._scaling_factor
        scale = 1 + delta * torch.sign(g)*diff
        return g * scale, None, None

class STE_discretizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_in, num_levels):
        x = x_in * (num_levels - 1)
        x = torch.round(x)
        x_out = x / (num_levels - 1)
        return x_out
    @staticmethod
    def backward(ctx, g):
        return g, None

# ref. https://github.com/ricky40403/DSQ/blob/master/DSQConv.py#L18
class QConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, args, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(QConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.quan_weight = args.QWeightFlag
        self.quan_act = args.QActFlag
        self.baseline = args.baseline
        self.STE_discretizer = STE_discretizer.apply
        self.EWGS_discretizer = EWGS_discretizer.apply
    
        if self.quan_weight:
            self.weight_levels = args.weight_levels
            self.uW = nn.Parameter(data = torch.tensor(0).float())
            self.lW = nn.Parameter(data = torch.tensor(0).float())
            self.register_buffer('bkwd_scaling_factorW', torch.tensor(args.bkwd_scaling_factorW).float())

        if self.quan_act:
            self.act_levels = args.act_levels
            self.uA = nn.Parameter(data = torch.tensor(0).float())
            self.lA = nn.Parameter(data = torch.tensor(0).float())
            self.register_buffer('bkwd_scaling_factorA', torch.tensor(args.bkwd_scaling_factorA).float())

        self.register_buffer('init', torch.tensor([0]))
        self.output_scale = nn.Parameter(data = torch.tensor(1).float())
        
        self.hook_Qvalues = False
        self.buff_weight = None
        self.buff_act = None

    def weight_quantization(self, weight):
        weight = (weight - self.lW) / (self.uW - self.lW)
        weight = weight.clamp(min=0, max=1) # [0, 1]

        if not self.baseline:
            weight = self.EWGS_discretizer(weight, self.weight_levels, self.bkwd_scaling_factorW)
        else:
            weight = self.STE_discretizer(weight, self.weight_levels)
            
        if self.hook_Qvalues:
            self.buff_weight = weight
            self.buff_weight.retain_grad()
        
        weight = (weight - 0.5) * 2 # [-1, 1]

        return weight

    def act_quantization(self, x):
        x = (x - self.lA) / (self.uA - self.lA)
        x = x.clamp(min=0, max=1) # [0, 1]
        
        if not self.baseline:
            x = self.EWGS_discretizer(x, self.act_levels, self.bkwd_scaling_factorA)
        else:
            x = self.STE_discretizer(x, self.act_levels)
            
        if self.hook_Qvalues:
            self.buff_act = x
            self.buff_act.retain_grad()

        return x

    def initialize(self, x):
        # self.init.data.fill_(0)
        Qweight = self.weight
        Qact = x
        
        if self.quan_weight:
            self.uW.data.fill_(self.weight.std()*3.0)
            self.lW.data.fill_(-self.weight.std()*3.0)
            Qweight = self.weight_quantization(self.weight)

        if self.quan_act:
            self.uA.data.fill_(x.std() / math.sqrt(1 - 2/math.pi) * 3.0)
            self.lA.data.fill_(x.min())
            Qact = self.act_quantization(x)

        Qout = F.conv2d(Qact, Qweight, self.bias,  self.stride, self.padding, self.dilation, self.groups)
        out = F.conv2d(x, self.weight, self.bias,  self.stride, self.padding, self.dilation, self.groups)
        self.output_scale.data.fill_(out.abs().mean() / Qout.abs().mean())


    def forward(self, x):
        if self.init == 1:
            self.initialize(x)
        
        Qweight = self.weight
        if self.quan_weight:
            Qweight = self.weight_quantization(Qweight)

        Qact = x
        if self.quan_act:
            Qact = self.act_quantization(Qact)

        output = F.conv2d(Qact, Qweight, self.bias,  self.stride, self.padding, self.dilation, self.groups) * torch.abs(self.output_scale)

        return output