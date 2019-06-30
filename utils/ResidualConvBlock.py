from torch import nn
import torch.nn.functional as F

class ResidualConvBlock(nn.Module):

    def __init__(self, c_in, c_out, k_size=3, stride=1, pad=1, bn=False, conv_layers=4, lrelu=True, conv_residual=False):
        nn.Module.__init__(self)
        self.c_in = c_in
        self.c_out = c_out
        self.k_size = k_size
        self.stride = stride
        self.pad = pad
        self.bn = bn
        self.conv_layers = conv_layers

        if bn:
            self.bns = nn.ModuleList([nn.BatchNorm2d(c_out) for i in range(conv_layers)])

        self.convs = nn.ModuleList([nn.Conv2d(c_in, c_out, k_size, stride=stride, padding=pad)])
        for i in range(conv_layers - 1):
            self.convs.append(nn.Conv2d(c_out, c_out, k_size, stride=stride, padding=pad))

        self.residual_conv = nn.Conv2d(c_in, c_out, 1)


    def forward(self, x):
        conv_out = x
        # conv_out = self.convs(conv_out)
        for layer_index in range(self.conv_layers):
            if self.bn:
                conv_out = self.bns[layer_index](conv_out)
            conv_out = self.convs[layer_index](F.relu(conv_out))

        return F.interpolate(self.residual_conv(x), scale_factor=1, mode='bilinear') + conv_out
