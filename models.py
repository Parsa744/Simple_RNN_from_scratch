import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.distributed.tensor.parallel import loss_parallel
from torch.onnx.symbolic_opset9 import tensor
import torch.optim as optim


def relu(x):
    """ReLU activation function using NumPy"""
    return np.maximum(0, x)


class npRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.init_weights()

    def init_weights(self):
        self.W_x = np.random.randn(self.hidden_size, self.input_size) * 0.1
        self.W_h = np.random.randn(self.hidden_size, self.hidden_size) * 0.1
        self.W_y = np.random.randn(self.output_size, self.hidden_size)
        self.bh = np.random.randn(self.hidden_size) * 0.1
        self.by = np.random.randn(self.output_size) * 0.1

    def forward(self, input, hidden):
        self.hidden = np.dot(self.W_x, input) + np.dot(self.W_h, hidden) + self.bh
        self.hidden = relu(self.hidden)
        self.output = np.dot(self.hidden,self.W_h) + self.by
        self.output = relu(self.output)
        return self.output , self.hidden

    def loss(self,y_pred,y_input):
        loss = (np.sum(y_input - y_pred)**2)
        return loss
    def back_prop(self,loss):
        pass
    def update_weights(self):
        pass


class torchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(torchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.init_weights()



    def init_weights(self):
        # Initialize as Parameter tensors, not nn.Linear layers
        self.W_x = nn.Parameter(torch.randn(self.hidden_size, self.input_size) * 0.1)
        self.W_h = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size) *0.1)  # What shape?
        self.W_y = nn.Parameter(torch.randn(self.output_size, self.hidden_size) *0.1)  # What shape?
        self.bh = nn.Parameter(torch.randn(self.hidden_size, 1) * 0.1)
        self.by = nn.Parameter(torch.randn(self.output_size, 1) *0.1)  # What shape?

    def forward(self, input, hidden):
        if not isinstance(hidden, torch.Tensor):
            hidden = torch.FloatTensor(data=hidden).unsqueeze(1)
        if not isinstance(input, torch.Tensor):
            input = torch.FloatTensor(data=input).unsqueeze(1)
        print(np.shape(self.W_x))
        print(np.shape(input))

        self.hidden = torch.mm(self.W_x, input) + torch.mm(self.W_h, hidden) + torch.FloatTensor(self.bh)
        self.hidden = torch.relu(self.hidden)
        self.output = torch.mm(self.W_h,self.hidden) + self.by
        self.output = torch.relu(self.output)
        return self.output , self.hidden

