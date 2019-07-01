from torch import nn
import torch.nn.functional as F

class ResidualDeconvBlock(nn.Module):
    """Discriminator containing 4 convolutional layers."""

    def __init__(self, c_in, c_out, k_size=4, stride=2, pad=1, bn=True, lrelu=True, conv_residual=False):
        nn.Module.__init__(self)
        self.c_in = c_in
        self.c_out = c_out
        self.k_size = k_size
        self.stride = stride
        self.pad = pad
        self.bn = bn


        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.deconv1 = nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad)
        self.deconv2 = nn.ConvTranspose2d(c_out, c_out, 3, 1, 1)

        self.deconvs = nn.Sequential(
            self.deconv1,
        )
        if bn:
            self.deconvs = nn.Sequential(
                self.deconvs,
                nn.BatchNorm2d(c_out)
            )

        self.deconvs = nn.Sequential(
            self.deconvs,
            nn.LeakyReLU(0.05),
            # nn.BatchNorm2d(c_out)
        )

        if bn:
            self.deconvs = nn.Sequential(
                self.deconvs,
                nn.BatchNorm2d(c_out)
            )

        self.residual_conv = nn.Conv2d(c_in, c_out, 1)


    def forward(self, x):

        out = self.deconvs(x)
        out = out + self.residual_conv(self.upsample(x))
        return out
