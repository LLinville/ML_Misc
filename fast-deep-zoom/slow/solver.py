import numpy as np
import torch
import torch.nn as nn
import torch.nn.modules as modules
from torch.autograd import Variable
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os
import tqdm
import scipy.ndimage as nd

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


def deprocess(image_np):
    image_np = image_np.squeeze().transpose(1, 2, 0)
    image_np = image_np * std.reshape((1, 1, 3)) + mean.reshape((1, 1, 3))
    image_np = np.clip(image_np, 0.0, 255.0)
    return image_np


def clip(image_tensor):
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[0, c] = torch.clamp(image_tensor[0, c], -m / s, (1 - m) / s)
    return image_tensor

scale_factor = 1.1

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.FloatTensor

class DeepZoomer:
    def __init__(self):
        self.scale_factor = 1.01
        self.frame_size = 256
        self.previous_depth = 1
        self.previous_frames = Variable(Tensor(np.random.random((self.previous_depth, 3, self.frame_size, self.frame_size))))
        self.target_layer = 15
        self.resize = modules.Upsample(size=(self.frame_size, self.frame_size), mode='bilinear')


        network = models.vgg19(pretrained=True)
        layers = list(network.features.children())
        self.model = nn.Sequential(*layers[: (self.target_layer + 1)])
        if torch.cuda.is_available:
            self.model = self.model.cuda()
        print(network)

    def loss(self, output):
        loss = 0
        for previous_frame in self.previous_frames:
            loss += (previous_frame - output).pow(2).sum()
        return loss

    def dream(self, iterations=45, lr=0.05):
        for i in range(iterations):
            self.model.zero_grad()
        """ Updates the image to maximize outputs for n iterations """
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.FloatTensor
        width_to_remove = int((1 - 1 / self.scale_factor) * self.frame_size / 2)
        self.previous_frames = modules.padding.F.pad(self.previous_frames,
                                                     (-1 * width_to_remove, -1 * width_to_remove, -1 * width_to_remove, -1 * width_to_remove))
        self.previous_frames = self.resize(self.previous_frames)
        frame = Variable(Tensor(self.previous_frames[-1]), requires_grad=True)


        frame = frame.view(1, 3, self.frame_size, self.frame_size)
        Variable(Tensor(frame), requires_grad=True)
        for i in range(iterations):
            self.model.zero_grad()
            out = self.model(frame)
            out_magnitude_loss = ((frame - 0.5)**2).sum()
            loss = out.norm() - 1*self.loss(frame) - 1*out_magnitude_loss
            frame.retain_grad()
            loss.backward()
            avg_grad = np.abs(frame.grad.data.cpu().numpy()).mean()
            norm_lr = lr / avg_grad
            frame.data += norm_lr * frame.grad.data
            frame.data = clip(frame.data)
            frame.grad.data.zero_()
        frame_out = frame.cpu().data.numpy()
        self.previous_frames = self.resize(self.previous_frames)
        self.previous_frames = torch.cat((self.previous_frames[:-1], Tensor(frame_out)))


if __name__ == "__main__":
    zoomer = DeepZoomer()
    plt.figure(figsize=(20, 20))

    for i in range(10000):
        for j in range(1):
            zoomer.dream()
        plt.imshow(zoomer.previous_frames[-1].cpu().permute(1, 2, 0))
        # plt.show()
        plt.pause(0.0001)
        plt.clf()





























