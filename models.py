import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.distributed.tensor.parallel import loss_parallel
from torch.onnx.symbolic_opset9 import tensor
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

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


        self.hidden = torch.mm(self.W_x, input) + torch.mm(self.W_h, hidden) + torch.FloatTensor(self.bh)
        self.hidden = torch.relu(self.hidden)
        self.output = torch.mm(self.W_h,self.hidden) + self.by
        self.output = torch.relu(self.output)
        return self.output , self.hidden



class seq2oneRNN(torchRNN):
    def __init__(self, input_size, hidden_size, output_size, seq_length):
        super(seq2oneRNN, self).__init__(input_size, hidden_size, output_size)
        self.seq_length = seq_length

    def forward_for_seq(self, input_seq, hidden):
        for input in input_seq:
            output, hidden = self.forward(input, hidden)
        return output, hidden

class one2seqRNN(torchRNN):
    def __init__(self, input_size, hidden_size, output_size, seq_length):
        super(one2seqRNN, self).__init__(input_size, hidden_size, output_size)
        self.seq_length = seq_length
    def forward_for_seq(self, input, hidden, seq_length):
        output = []
        for i in range(seq_length):
            output, hidden = self.forward(input, hidden)
            output.append(output)
        return output, hidden

class one2oneRNN(torchRNN):
    def __init__(self, input_size, hidden_size, output_size, recurrent_length):
        super(one2oneRNN, self).__init__(input_size, hidden_size, output_size)
        self.recurrent_length = recurrent_length
    def forward_for_seq(self, input, hidden):
        for i in range(self.recurrent_length):
            output, hidden = self.forward(input, hidden)
        return output, hidden



class seq2seqRNN(torchRNN):
    def __init__(self, input_size, hidden_size, output_size, seq_length):
        super(seq2seqRNN, self).__init__(input_size, hidden_size, output_size)
        self.seq_length = seq_length
    def forward_for_seq(self, input_seq, hidden):
        output = []
        for input in input_seq:
            New_output, hidden = self.forward(input, hidden)
            output.append(New_output)
        return output, hidden


def EncodeDecoderRNN(input_seq, hidden_for_Encoder, hidden_for_Decoder,hidden_size,recurrent_length,output_len):
    Encoder = seq2oneRNN(len(input_seq),hidden_size,output_len)
    encoder_outputs, _ = Encoder.forward_for_seq(input_seq, hidden_for_Encoder)
    Decoder = one2oneRNN(len(encoder_outputs),hidden_size,output_len,recurrent_length)
    output, _ = Decoder.forward_for_seq(encoder_outputs[-1],hidden_for_Decoder)
    return output

def EncoderDecoderExample():
    x1 = [1,2,2]
    Y1 = [0,2,0,4]
    H0 = [0,0,0,0]
    x2 = [1,2,4]
    Y2 = [3,0,1,1]
    x3 = [3,0,5]
    Y3 = [5,2,7,3]
    x4 = [5,6,7]
    Y4 = [3,4,9,2]
    result = EncodeDecoderRNN(input_seq=[x1,x2,x3,x4],hidden_for_Encoder=H0,hidden_for_Decoder=H0,hidden_size=4,recurrent_length=4,output_len=4)

def seq2seqExaple():
    x1 = [1,2,2]
    Y1 = [0,2,0,4]
    H0 = [0,0,0,0]
    x2 = [1,2,4]
    Y2 = [3,0,1,1]
    x3 = [3,0,5]
    Y3 = [5,2,7,3]
    x4 = [5,6,7]
    Y4 = [3,4,9,2]
    myRNN = seq2seqRNN(3,4,4,4)
    mySeq = [x1,x2,x3,x4]
    myYseq = [Y1,Y2,Y3,Y4]
    Yp1 , _ = myRNN.forward_for_seq(mySeq,H0)
    for name, param in myRNN.named_parameters():
        print(f"Parameter name: {name}")
        print(f"Parameter shape: {param.shape}")
        print(f"Parameter values:\n{param}\n")
    for name, param in myRNN.named_parameters():
        if param.dim() > 1:  # Only plot 2D weights, skip biases
            plt.figure(figsize=(6, 4))
            sns.heatmap(param.detach().cpu().numpy(), annot=True, cmap='viridis')
            plt.title(f"Heatmap of {name}")
            plt.xlabel("Output features")
            plt.ylabel("Input features")
            plt.show()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(myRNN.parameters(), lr=0.1)
    total_loss = 0
    for index in range(len(Yp1)):

        optimizer.zero_grad()
        prediction = torch.FloatTensor(Yp1[index]).unsqueeze(1)
        gt = torch.FloatTensor(myYseq[index]).unsqueeze(1)
        loss = criterion(prediction, gt)
        print(loss)
        total_loss+=loss
    loss.backward()
    optimizer.step()
    print('total_loss',total_loss)
    for name, param in myRNN.named_parameters():
        print(f"Parameter name: {name}")
        print(f"Parameter shape: {param.shape}")
        print(f"Parameter values:\n{param}\n")
    for name, param in myRNN.named_parameters():
        if param.dim() > 1:  # Only plot 2D weights, skip biases
            plt.figure(figsize=(6, 4))
            sns.heatmap(param.detach().cpu().numpy(), annot=True, cmap='viridis')
            plt.title(f"Heatmap of {name}")
            plt.xlabel("Output features")
            plt.ylabel("Input features")
            plt.show()

def main():
    seq2seqExaple()
    return 0



if __name__ == "__main__":
    main()