"""Convolution using native tensor operations, avoiding MPS conv backward.

All tensor arithmetic and gradients stay on the input device. This trades
optimized convolution speed for a path built from padding, slices, matmul,
and additions. CPU/CUDA keep their normal convolution implementation.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair


def tensor_conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if x.device != weight.device or (bias is not None and bias.device != x.device):
        raise RuntimeError('Convolution tensors must share one device')
    sh, sw = _pair(stride)
    dh, dw = _pair(dilation)
    ph, pw = _pair(padding)
    kh, kw = weight.shape[-2:]
    if ph or pw:
        x = F.pad(x, (pw, pw, ph, ph))
    oh = (x.shape[2] - dh * (kh - 1) - 1) // sh + 1
    ow = (x.shape[3] - dw * (kw - 1) - 1) // sw + 1
    if oh <= 0 or ow <= 0:
        raise ValueError('Kernel is larger than padded input')
    cin = x.shape[1] // groups
    cout = weight.shape[0] // groups
    outputs = []
    for group in range(groups):
        value = None
        for ky in range(kh):
            for kx in range(kw):
                patch = x[:, group*cin:(group+1)*cin,
                          ky*dh:ky*dh+oh*sh:sh,
                          kx*dw:kx*dw+ow*sw:sw]
                # [N,H,W,Cin] @ [Cin,Cout], including non-square/grouped kernels.
                kernel = weight[group*cout:(group+1)*cout, :, ky, kx]
                part = torch.matmul(patch.permute(0, 2, 3, 1).contiguous(), kernel.t().contiguous())
                value = part if value is None else value + part
        outputs.append(value.permute(0, 3, 1, 2))
    output = torch.cat(outputs, dim=1) if groups > 1 else outputs[0]
    if bias is not None:
        output = output + bias.view(1, -1, 1, 1)
    return output.contiguous()


class MPSTensorConv2d(nn.Conv2d):
    def forward(self, x):
        if x.device.type != 'mps' or not torch.is_grad_enabled():
            return self._conv_forward(x, self.weight, self.bias)
        padding = self.padding
        if self.padding_mode != 'zeros':
            x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = (0, 0)
        return tensor_conv2d(x, self.weight, self.bias, self.stride,
                             padding, self.dilation, self.groups)


def enable_mps_tensor_convolution(model):
    for module in model.modules():
        # Also replace the earlier CPU-gradient workaround if installed.
        if isinstance(module, nn.Conv2d) and type(module) is not MPSTensorConv2d:
            module.__class__ = MPSTensorConv2d
