import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable




class Net(nn.Module):
    def __init__(self, input_size, steps_to_unroll, target):
        super(Net, self).__init__()
        self.state_depth = input_size[0]
        self.steps_to_unroll = steps_to_unroll
        self.target = torch.Tensor(target).cuda()
        self.conv1_depth = 3 * self.state_depth
        self.sobelx = torch.Tensor(np.array([
            [-1,0,1],
            [-2,0,2],
            [-1,0,1]
        ], dtype=np.float32))[np.newaxis,...]
        self.sobelx = np.repeat(self.sobelx[np.newaxis,...], self.state_depth, 0)
        self.sobely = self.sobelx.transpose(2,3).cuda()
        self.sobelx = self.sobelx.cuda()
        self.zero = torch.Tensor([0.0, 0.0, 0.0]).cuda()
        self.conv1 = nn.Conv2d(self.conv1_depth, self.state_depth, (1,1))
        self.conv2 = nn.Conv2d(self.state_depth, self.state_depth, (1,1))
        self.relu = nn.ReLU()

    def forward(self, state):
        out = state
        for step in range(self.steps_to_unroll):
            identity = out
            sobelx_state = torch.zeros_like(out)
            sobelx_state[...,1:-1,1:-1] = F.conv2d(out, self.sobelx, self.zero, stride=1, groups=self.state_depth)
            sobely_state = torch.zeros_like(out)
            sobely_state[..., 1:-1, 1:-1] = F.conv2d(out, weight=self.sobely, bias=self.zero, stride=1, groups=self.state_depth)
            out = torch.cat((sobelx_state, sobely_state, out), dim=1)
            out = self.conv1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = identity + F.tanh(out)
        return out

learning_rate = 0.005
num_epochs = 10000

fire_rate = 0.5
depth = 3
grid_size = depth, 6, 6
num_grids = 16

target = np.array([[0 if np.sqrt(x*x + y*y) > 2 else 1 for x in range(-3, 3)] for y in range(-3, 3)])
# target = np.repeat(target[np.newaxis,...], num_grids, 0)

starting = np.zeros([*grid_size], dtype=np.float32)
starting[...,grid_size[0]//2-1:grid_size[0]//2+1, grid_size[1]//2-1:grid_size[1]//2+1] = 1
print(starting)


models = [Net(grid_size, 5, target).cuda() for i in range(num_grids)]


optimizers = [torch.optim.Adam(model.parameters(), lr=learning_rate) for model in models]

# Train the Model
for epoch in range(num_epochs):
    # Convert torch tensor to Variable
    grid = Variable(torch.Tensor(starting)).cuda()
    target_grids= Variable(torch.Tensor(target)).cuda()

    # Forward + Backward + Optimize
    optimizer.zero_grad()  # zero the gradient buffer
    outputs = model(grids)
    loss = torch.sum(torch.pairwise_distance(outputs, target_grids[np.newaxis,...]))
    loss = loss*loss
    loss.backward()
    optimizer.step()

    print('Epoch [%d/%d], Loss: %.4f'
          % (epoch + 1, num_epochs, loss.data.item()))
