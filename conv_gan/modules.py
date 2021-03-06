import torch.nn as nn
import torch.nn.functional as F
from utils.plotter.plotter import Plotter
from utils.ResidualDeconvBlock import ResidualDeconvBlock
from utils.ResidualConvBlock import ResidualConvBlock
from utils.SummingResidualDeconvBlock import SummingResidualDeconvBlock
import torch


def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom convolutional layer for simplicity."""
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


class Discriminator_Vanilla(nn.Module):
    """Discriminator containing 4 convolutional layers."""

    def __init__(self, image_size=128, conv_dim=32):
        super(Discriminator_Vanilla, self).__init__()
        self.conv1 = conv(3, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)
        self.conv4 = conv(conv_dim * 4, conv_dim * 4, 4)
        self.conv5 = conv(conv_dim * 4, conv_dim * 8, 4)
        self.fc = conv(conv_dim * 8, 1, int(image_size / 32), 1, 0, False)

    def forward(self, x):  # If image_size is 64, output shape is as below.

        out = F.leaky_relu(self.conv1(x), 0.05)  # (?, 64, 32, 32)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 16, 16)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 8, 8)
        out = F.leaky_relu(self.conv4(out), 0.05)  # (?, 512, 4, 4)
        out = F.leaky_relu(self.conv5(out), 0.05)
        out = self.fc(out).squeeze()
        return out


class Discriminator(nn.Module):
    """Discriminator containing 4 convolutional layers."""

    def __init__(self, image_size=128, conv_dim=32):
        self.plotter = Plotter()
        self.iter = 0
        super(Discriminator, self).__init__()
        # self.conv1 = conv()

        self.conv1 = nn.Sequential(
            conv(3, conv_dim, 4, bn=False))  # , conv(conv_dim, conv_dim * 2, 1, stride=1, pad=0))
        # self.conv2 = nn.Sequential(conv(conv_dim*2, conv_dim*2, 4, bn=False), conv(conv_dim * 2, conv_dim * 4, 1, stride=1, pad=0))
        # self.conv3 = nn.Sequential(conv(conv_dim*4, conv_dim*4, 4, bn=False), conv(conv_dim * 4, conv_dim * 8, 1, stride=1, pad=0))
        # self.conv4 = nn.Sequential(conv(conv_dim*8, conv_dim*8, 4, bn=False), conv(conv_dim * 8, conv_dim * 16, 1, stride=1, pad=0))
        # self.conv5 = nn.Sequential(conv(conv_dim*16, conv_dim*16, 4, bn=False), conv(conv_dim * 16, conv_dim * 32, 1, stride=1, pad=0))
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)
        self.conv4 = conv(conv_dim * 4, conv_dim * 8, 4)
        self.conv5 = conv(conv_dim * 8, conv_dim * 16, 4)
        # self.fc = conv(conv_dim * 16, 1, conv_dim * 16, 1, 0, False)

        self.fc1 = nn.Linear(conv_dim * 16 * 4 * 2, 32)
        # self.fc1 = nn.Linear(4608, 1)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):  # If image_size is 64, output shape is as below.

        out = F.leaky_relu(self.conv1(x), 0.05)  # (?, 64, 32, 32)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 16, 16)
        self.iter += 1
        # print(self.iter)
        # if self.iter % 101 == 0:
        #     self.plotter.draw_activations(out, original=x)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 8, 8)
        out = F.leaky_relu(self.conv4(out), 0.05)  # (?, 512, 4, 4)
        # out = F.leaky_relu(self.conv5(out), 0.05)
        flattened = out.view(out.size(0), -1)
        out = self.fc1(flattened).squeeze()
        out = self.fc2(out)
        return out


class Critic(nn.Module):
    # Wasserstein critic, no minimum or maximum loss

    def __init__(self):
        super(Critic, self).__init__()

        base_features = 64
        image_size = 64
        self.residual_downconv = nn.Sequential(
            ResidualConvBlock(3, base_features), #, k_size=4, stride=2, pad=1, bn=False),
            nn.AdaptiveAvgPool2d((image_size, image_size//2)),
            ResidualConvBlock(base_features, base_features*2),#, k_size=4, stride=2, pad=1),
            nn.AdaptiveAvgPool2d((image_size//2, image_size//4)),
            ResidualConvBlock(base_features*2, base_features*4),#, k_size=4, stride=2, pad=1),
            nn.AdaptiveAvgPool2d((image_size//4, image_size//8)),
            # ResidualConvBlock(base_features*4, base_features*8),#, k_size=4, stride=2, pad=1),
            # nn.AdaptiveAvgPool2d(base_features*8)
        )

        self.fc1 = nn.Linear(32768, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.LeakyReLU(0.05)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.relu(self.residual_downconv(x))
        flattened = out.view(out.size(0), -1)
        return self.sigmoid(self.fc2(self.relu(self.fc1(flattened))))

class ResidualGenerator(nn.Module):
    """Generator containing 7 deconvolutional layers."""

    def __init__(self, z_dim=256):
        super(ResidualGenerator, self).__init__()

        self.startingDimension = 4, 4
        self.startingFeatures = 512
        self.fc = nn.Linear(z_dim, self.startingDimension[0] * self.startingDimension[1] * self.startingFeatures)

        self.residual_deconv = nn.Sequential(
            ResidualDeconvBlock(512, 256),
            ResidualDeconvBlock(256, 128),
            ResidualDeconvBlock(128, 64),
            ResidualDeconvBlock(64, 32, bn=False)
        )

        self.toColorChannels = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, padding=0),
            nn.Tanh()
        )

    def forward(self, z):
        # z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.fc(z)
        numStartingFeatures = 512
        startingDimensions = 4, 4
        try:
            out = out.view(32, numStartingFeatures, *startingDimensions)
        except Exception as e:
            print(e)

        out = self.residual_deconv(out)
        out = self.toColorChannels(out)
        return out

    # def forward(self):
    #
    #
    #     # self.upsize = nn.Sequential(*[
    #     #     deconv(512, 256),
    #     #     deconv(256, 128),
    #     #     deconv(128, 64),
    #     #     nn.Upsample(64, scale_factor=2),
    #     #     nn.Conv2d(64, 3, 3, padding=1),
    #     #     nn.Tanh()
    #     # ])
    #
    #     # self.spread1 = nn.Upsample(scale_factor=2, mode='bilinear')
    #     # self.conv1 = nn.Conv2d(512, 256, 3, padding=1)
    #     # self.bn1 = nn.BatchNorm2d(256)
    #     # self.spread2 = nn.Upsample(scale_factor=2, mode='bilinear')
    #     # self.conv2 = nn.Conv2d(256, 128, 3, padding=1)
    #     # self.bn2 = nn.BatchNorm2d(128)
    #     # self.spread3 = nn.Upsample(scale_factor=2, mode='bilinear')
    #     # self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
    #     # self.bn3 = nn.BatchNorm2d(64)
    #     # self.spread4 = nn.Upsample(scale_factor=2, mode='bilinear')
    #     # self.conv4 = nn.Conv2d(64, 3, 3, padding=1)
    #     # self.conv1 = deconv(conv_dim*8, conv_dim*4, 4)
    #     # self.conv2 = deconv(conv_dim*4, conv_dim*2, 4)
    #     # self.conv3 = deconv(conv_dim*2, conv_dim, 4)`
    #     # # self.conv4 = deconv(conv_dim, 32, 2, bn=True)
    #     # self.conv4 = deconv_vanilla(conv_dim, 32, 3, stride=1, pad=0)
    #     # self.to_rgb = deconv_vanilla(32, 3, 1, stride=1, pad=0, bn=False)
    #     # self.upsize = nn.Sequential(self.conv1, self.conv2, self.conv3,
    #     #                             # nn.Upsample(scale_factor=2, mode='bilinear'),
    #     #                             deconv_vanilla(conv_dim, conv_dim, 32, stride=1, pad=1),
    #     #                             self.conv4, self.to_rgb)
    #     # sizes = [4, 16,  64, 128]
    #     # features = [512, 256, 128, 64, 32]
    #     # self.upsize = deconv_vanilla_chain(features=features)
    #     # self.upsize = nn.Sequential(
    #     #     deconv_vanilla(512, 256, k_size=4, stride=2, pad=2),
    #     #     deconv_vanilla(256, 128, k_size=6, stride=2, pad=1),
    #     #     deconv_vanilla(128, 128, k_size=6, stride=2, pad=2),
    #     #     deconv_vanilla(128, 64, k_size=6, stride=2, pad=2),
    #     #     deconv_vanilla(64, 32, k_size=3, stride=1, pad=1),
    #     #     deconv_vanilla(32, 16, k_size=5, stride=1, pad=1)
    #     # )
    #     # self.upsize = nn.Sequential(
    #     #     deconv_vanilla(512, 256, k_size=4, stride=2, pad=2),
    #     #     deconv_vanilla(256, 128, k_size=4, stride=2, pad=2),
    #     #     deconv_vanilla(128, 64, k_size=4, stride=2, pad=2),
    #     #     deconv_vanilla(64, 64, k_size=4, stride=2, pad=2),
    #     #     deconv_vanilla(64, 32, k_size=4, stride=2, pad=2),
    #     #     deconv_vanilla(32, 16, k_size=3, stride=1, pad=1, bn=False),
    #     # )
    #     self.toColorChannels = nn.Sequential(
    #         nn.Conv2d(32, 3, kernel_size=1, padding=0),
    #         nn.Tanh()
    #     )
    #
    # def forward(self, z):
    #     # z = z.view(z.size(0), z.size(1), 1, 1)
    #     out = self.fc(z)
    #     numStartingFeatures = 512
    #     startingDimensions = 4, 4
    #     try:
    #         out = out.view(64, numStartingFeatures, *startingDimensions)
    #     except Exception as e:
    #         print(e)
    #
    #     out = self.residual_deconv(out)
    #     out = self.toColorChannels(out)
    #     return out

class ResidualSumGenerator(nn.Module):
    """Generator containing 7 deconvolutional layers."""

    def __init__(self, z_dim=256):
        super(ResidualSumGenerator, self).__init__()

        self.startingDimension = 4, 4
        self.startingFeatures = 512
        self.n_blocks = 4
        self.batch_size = 32

        self.residuals = torch.FloatTensor([]).cuda()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.fc = nn.Linear(z_dim, self.startingDimension[0] * self.startingDimension[1] * self.startingFeatures)

        residual_sizes = [256, 384, 448, 480, 496]
        # residual_sizes = [0] * 10
        # residual_sizes = [512, 768, 896, 960]

        self.conv_modules = [nn.Sequential(
            nn.ConvTranspose2d(self.startingFeatures // (2 ** (n+1)),
                               self.startingFeatures // (2 ** (n+1)),
                               kernel_size=3, stride=1, padding=1
                               ).cuda(),
            nn.LeakyReLU(0.05))
            # nn.ConvTranspose2d(self.startingFeatures // (2 ** (n + 1)),
            #                    self.startingFeatures // (2 ** (n + 1)),
            #                    kernel_size=3, stride=1, padding=1
            #                    ).cuda(),
            # nn.LeakyReLU(0.05)
            # )
            for n in range(self.n_blocks)
        ]
        self.conv_modules = nn.Sequential(*self.conv_modules)
        self.blocks = [
            SummingResidualDeconvBlock(self.startingFeatures//(2**n),
                                       self.startingFeatures//(2**(n+1)),
                                       size=self.startingDimension[0]*(2**(n+1)),
                                       residual_size=residual_sizes[n],
                                       conv_module=self.conv_modules[n],
                                       bn= n!=self.n_blocks-1)
            for n in range(self.n_blocks)
        ]

        self.toColorChannels = self.toColorChannels = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, padding=0),
            nn.Tanh()
        )

    def forward(self, z):
        self.residuals = torch.FloatTensor([]).cuda()
        # z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.fc(z)
        try:
            out = out.view(self.batch_size, self.startingFeatures, *self.startingDimension)
        except Exception as e:
            print(e)

        # self.residuals = torch.cat((self.residuals, out), dim=1)

        for n, block in enumerate(self.blocks):
            # self.residuals = torch.cat((self.residuals, out), dim=1)
            out, self.residuals = block.forward(self.residuals, out)
        out = self.toColorChannels(out)
        return out