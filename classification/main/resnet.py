import torch.nn as nn
import torch
import torchvision
import numpy as np
from gate import GateModule192

def kaiming_init(module, a=0, mode='fan_out', nonlinearity='relu', bias=0, distribution='normal'):
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def constant_init(module, val, bias=0):
    nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class ResNetDCT_Upscaled_Static(nn.Module):
    def __init__(self, channels=24, pretrained=True, inputgate=False, inputmix=False):
        super(ResNetDCT_Upscaled_Static, self).__init__()

        self.input_gate = inputgate
        self.input_mix = inputmix

        model = torchvision.models.resnet50(pretrained = pretrained)

        self.model = nn.Sequential(*list(model.children())[4:-1])
        self.fc = list(model.children())[-1]
        self.relu = nn.ReLU(inplace=True)

        if channels == 0 or channels == 192:
            out_ch = self.model[0][0].conv1.out_channels
            self.model[0][0].conv1 = nn.Conv2d(channels, out_ch, kernel_size=3, stride=1, bias=False)
            kaiming_init(self.model[0][0].conv1)

            out_ch = self.model[0][0].downsample[0].out_channels
            self.model[0][0].downsample[0] = nn.Conv2d(channels, out_ch, kernel_size=3, stride=1, bias=False)
            kaiming_init(self.model[0][0].downsample[0])

        elif channels < 64:
            out_ch = self.model[0][0].conv1.out_channels
            temp_layer = nn.Conv2d(channels, out_ch, kernel_size=3, stride=1, bias=False)
            temp_layer.weight.data = self.model[0][0].conv1.weight.data[:, :channels]
            self.model[0][0].conv1 = temp_layer

            out_ch = self.model[0][0].downsample[0].out_channels
            temp_layer = nn.Conv2d(channels, out_ch, kernel_size=3, stride=1, bias=False)
            temp_layer.weight.data = self.model[0][0].downsample[0].weight.data[:, :channels]
            self.model[0][0].downsample[0] = temp_layer

        if inputmix:
            self.inputmix_layer = nn.Conv2d(192, channels, kernel_size=3, stride=1, bias=False)
            kaiming_init(self.inputmix_layer)

        if inputgate:
            self.inp_GM = GateModule192(channels=channels)
            self._initialize_weights()

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if 'inp_gate_l' in str(name):
                m.weight.data.normal_(0, 0.001)
                m.bias.data[::2].fill_(0.1)
                m.bias.data[1::2].fill_(2)
            elif 'inp_gate' in str(name):
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)

    def forward(self, x):
        if self.input_mix:
            x = self.inputmix_layer(x)

        if self.input_gate:
            x, inp_atten = self.inp_GM(x)

        x = self.model(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        if self.input_gate:
            return x, inp_atten
        else:
            return x