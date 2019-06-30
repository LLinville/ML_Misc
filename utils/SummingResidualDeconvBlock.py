from torch import nn
import torch.nn.functional as F
import torch

class SummingResidualDeconvBlock(nn.Module):
    """Discriminator containing 4 convolutional layers."""

    def __init__(self, c_in, c_out, k_size=4, stride=2, pad=1, size=32, residual_size=2, bn=False, residuals=None, conv_module=nn.Identity(), lrelu=True, conv_residual=False):
        nn.Module.__init__(self)
        self.residuals = residuals
        if not residuals:
            self.residuals = []
        self.c_in = c_in
        self.c_out = c_out
        self.k_size = k_size
        self.stride = stride
        self.pad = pad
        self.bn = nn.BatchNorm2d(size*2) if bn else nn.Identity()
        self.conv_module = conv_module
        self.residual_size = residual_size

        self.upsample = nn.Upsample(size=size, mode='bilinear')
        self.deconv = nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad).cuda()
        # residual_size = sum()
        if residual_size > 0:
            self.residual_conv = nn.Conv2d(residual_size, c_out, 1, stride=1, padding=0).cuda()
        else:
            self.residual_conv = nn.Conv2d(c_in, c_out, 1, stride=1, padding=0).cuda()

    def forward(self, residuals, x):
        x_upsampled = self.upsample(x)
        x_conved = self.conv_module(x)
        after_conv_deconved = self.bn(F.leaky_relu(self.deconv(x_conved), 0.05))
        upsampled_residuals = self.upsample(torch.cat((residuals, x), dim=1))
        residuals_after_conv = nn.LeakyReLU(self.residual_conv(upsampled_residuals), 0.05)
        # return after_conv_deconved + residuals_after_conv, upsampled_residuals

        return after_conv_deconved, upsampled_residuals