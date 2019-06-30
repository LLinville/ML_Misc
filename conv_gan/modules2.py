import torch.nn as nn
from utils.SummingResidualDeconvBlock import SummingResidualDeconvBlock
from utils.ResidualConvBlock import ResidualConvBlock

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

    def forward(self, x):
        out = self.residual_downconv(x)
        flattened = out.view(out.size(0), -1)
        return self.fc2(self.fc1(flattened))

class ResidualSumGenerator(nn.Module):
    """Generator containing 7 deconvolutional layers."""

    def __init__(self, z_dim=256):
        super(ResidualSumGenerator, self).__init__()

        self.startingDimension = 4, 4
        self.startingFeatures = 512
        self.n_blocks = 4

        self.blocks = [
            SummingResidualDeconvBlock(self.startingFeatures*n, self.startingFeatures*n*2)
            for n in range(self.n_blocks)
        ]

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
