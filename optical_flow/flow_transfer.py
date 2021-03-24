
import numpy as np
from imageio import imread
from torch.autograd import Variable
import torch

im1 = np.zeros((3, 100, 100))
im2 = np.asarray(imread("0002.png"))
im1 = np.float64(im1 / 255)
im2 = np.float64(im2 / 255)

desired_flow = np.zeros((2, 100, 100))
output_image = torch.rand((3, 100, 100), requires_grad=True)
flow_loss = desired_flow - cv2.calcOpticalFlowFarneback(prvs, next, 0.5, 3, 15, 3, 5, 1.2, 0)
loss =

desired_flow[..., 30:60] = 1