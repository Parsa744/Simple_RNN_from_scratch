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







def main():
    x1 = [1,2,2]
    Y1 = [0,2,0,4]
    H0 = [0,0,0,0]
    x2 = [1,2,4]
    Y2 = [3,0,1,1]
    x3 = [3,0,5]
    Y3 = [5,2,7,3]
    x4 = [5,6,7]
    Y4 = [3,4,9,2]
    myRNN = torchRNN(3,4,4)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(myRNN.parameters(), lr=0.1)

    Yp1,H1 = myRNN.forward(input=x1,hidden=H0)

    loss = criterion(Yp1,torch.FloatTensor(data=Y1).unsqueeze(1))
    optimizer.zero_grad()

    print('loss',loss)

    Yp2,H2 = myRNN.forward(input=x2,hidden=H1)


    loss = criterion(Yp2,torch.FloatTensor(data=Y2).unsqueeze(1))

    print('loss',loss)

    Yp3,H3 = myRNN.forward(input=x3,hidden=H2)

    loss = criterion(Yp3,torch.FloatTensor(data=Y3).unsqueeze(1))

    print('loss',loss)


    Yp4,H4 = myRNN.forward(input=x4,hidden=H3)


    loss = criterion(Yp4,torch.FloatTensor(data=Y4).unsqueeze(1))
    loss.backward()
    optimizer.step()
    print('loss',loss)



if __name__ == "__main__":
    main()