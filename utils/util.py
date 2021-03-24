from os import path
from torch import nn

def first_unoccupied(pattern):
    for i in range(1, 10000):
        if not path.exists(pattern % i):
            return pattern % i
    return None

class Reshape_Layer(nn.Module):
    def __init__(self, *args):
        super(Reshape_Layer, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

