import torch.nn as nn
import torch.nn.functional as F
from plotter.plotter import Plotter
from ResidualDeconvBlock import ResidualDeconvBlock


def deconv_vanilla(c_in, c_out, k_size, stride=2, pad=1, bn=True, lrelu=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    if lrelu:
        layers.append(nn.LeakyReLU(0.05))
    return nn.Sequential(*layers)


def deconv(c_in, c_out, conv_count=2, activation=nn.ReLU, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(nn.Upsample(mode='bilinear', scale_factor=2))
    layers.append(nn.ConvTranspose2d(c_in, c_out, 1))
    for conv_index in range(conv_count):
        layers.append(nn.ConvTranspose2d(c_out if conv_index == 0 else c_out, c_out, 3, padding=1))
        if bn:
            layers.append(nn.BatchNorm2d(c_out))
        layers.append(nn.LeakyReLU(0.05))
    # layers.append(nn.Upsample(c_in))#, mode='bilinear', scale_factor=2 ))
    # for conv_index in range(conv_count):
    #     layers.append(nn.Conv2d(c_in if conv_index == 0 else c_out, c_out, 3, padding=1))
    #     layers.append(nn.LeakyReLU(0.05))
    #     if bn:
    #         layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


def residual_deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True, lrelu=True):
    layers = []
    nn.Sequential()


def deconv_vanilla_chain(features, kernelSize=4):
    layers = []

    for index in range(len(features) - 1):
        layers.append(
            nn.Sequential(
                deconv_vanilla(features[index], features[index] // 2, k_size=1, stride=1, pad=0),
                nn.LeakyReLU(0.05),
                deconv_vanilla(features[index] // 2, features[index] // 2, kernelSize),
                nn.LeakyReLU(0.05),
            )
        )

    return nn.Sequential(*layers)


class Generator(nn.Module):
    """Generator containing 7 deconvolutional layers."""

    def __init__(self, z_dim=256, image_size=128, conv_dim=64):
        super(Generator, self).__init__()

        self.startingDimension = 4, 4
        self.startingFeatures = 512
        self.fc = nn.Linear(z_dim, self.startingDimension[0] * self.startingDimension[1] * self.startingFeatures)

        self.residual_deconv = nn.Sequential(
            ResidualDeconvBlock(512, 256),
            ResidualDeconvBlock(256, 128),
            ResidualDeconvBlock(128, 64),
            ResidualDeconvBlock(64, 32, bn=False)
        )

        # self.upsize = nn.Sequential(*[
        #     deconv(512, 256),
        #     deconv(256, 128),
        #     deconv(128, 64),
        #     nn.Upsample(64, scale_factor=2),
        #     nn.Conv2d(64, 3, 3, padding=1),
        #     nn.Tanh()
        # ])

        # self.spread1 = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.conv1 = nn.Conv2d(512, 256, 3, padding=1)
        # self.bn1 = nn.BatchNorm2d(256)
        # self.spread2 = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.conv2 = nn.Conv2d(256, 128, 3, padding=1)
        # self.bn2 = nn.BatchNorm2d(128)
        # self.spread3 = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
        # self.bn3 = nn.BatchNorm2d(64)
        # self.spread4 = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.conv4 = nn.Conv2d(64, 3, 3, padding=1)
        # self.conv1 = deconv(conv_dim*8, conv_dim*4, 4)
        # self.conv2 = deconv(conv_dim*4, conv_dim*2, 4)
        # self.conv3 = deconv(conv_dim*2, conv_dim, 4)`
        # # self.conv4 = deconv(conv_dim, 32, 2, bn=True)
        # self.conv4 = deconv_vanilla(conv_dim, 32, 3, stride=1, pad=0)
        # self.to_rgb = deconv_vanilla(32, 3, 1, stride=1, pad=0, bn=False)
        # self.upsize = nn.Sequential(self.conv1, self.conv2, self.conv3,
        #                             # nn.Upsample(scale_factor=2, mode='bilinear'),
        #                             deconv_vanilla(conv_dim, conv_dim, 32, stride=1, pad=1),
        #                             self.conv4, self.to_rgb)
        # sizes = [4, 16,  64, 128]
        # features = [512, 256, 128, 64, 32]
        # self.upsize = deconv_vanilla_chain(features=features)
        # self.upsize = nn.Sequential(
        #     deconv_vanilla(512, 256, k_size=4, stride=2, pad=2),
        #     deconv_vanilla(256, 128, k_size=6, stride=2, pad=1),
        #     deconv_vanilla(128, 128, k_size=6, stride=2, pad=2),
        #     deconv_vanilla(128, 64, k_size=6, stride=2, pad=2),
        #     deconv_vanilla(64, 32, k_size=3, stride=1, pad=1),
        #     deconv_vanilla(32, 16, k_size=5, stride=1, pad=1)
        # )
        # self.upsize = nn.Sequential(
        #     deconv_vanilla(512, 256, k_size=4, stride=2, pad=2),
        #     deconv_vanilla(256, 128, k_size=4, stride=2, pad=2),
        #     deconv_vanilla(128, 64, k_size=4, stride=2, pad=2),
        #     deconv_vanilla(64, 64, k_size=4, stride=2, pad=2),
        #     deconv_vanilla(64, 32, k_size=4, stride=2, pad=2),
        #     deconv_vanilla(32, 16, k_size=3, stride=1, pad=1, bn=False),
        # )
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
            out = out.view(64, numStartingFeatures, *startingDimensions)
        except Exception as e:
            print(e)

        out = self.residual_deconv(out)
        out = self.toColorChannels(out)
        return out


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
        super(Discriminator, self).__init__()
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

    def __init__(self, image_size=128, conv_dim=16):
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





