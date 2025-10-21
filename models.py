import numpy as np
import torch
import torch.nn as nn



class torchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(torchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.W_x_and_h_layer = nn.Linear(self.input_size + self.hidden_size, self.hidden_size, bias=True)
        self.W_y_layer = nn.Linear(self.hidden_size,self.output_size,bias=True)

    def forward(self, input, hidden):
        if not isinstance(hidden, torch.Tensor):
            hidden = torch.FloatTensor(data=hidden).unsqueeze(1)
        if not isinstance(input, torch.Tensor):
            input = torch.FloatTensor(data=input).unsqueeze(1)


        self.hidden = self.W_x_and_h_layer(torch.transpose(torch.concat([input,hidden],dim=0),1,0))
        self.hidden = torch.relu(self.hidden)
        self.output = self.W_y_layer(self.hidden)
        self.output = torch.relu(self.output)
        self.hidden = torch.transpose(self.hidden,1,0)
        return self.output , self.hidden



class seq2oneRNN(torchRNN):
    def __init__(self, input_size, hidden_size, output_size):
        super(seq2oneRNN, self).__init__(input_size, hidden_size, output_size)

    def forward_for_seq(self, input_seq, hidden):

        for input in input_seq:
            output, hidden = self.forward(torch.FloatTensor(data=input).unsqueeze(1),hidden)
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
    print(len(input_seq),hidden_size,output_len)
    print(hidden_for_Encoder,hidden_for_Decoder)
    print(input_seq)
    Encoder = seq2oneRNN(len(input_seq),hidden_size,output_len)
    encoder_outputs, _ = Encoder.forward_for_seq(input_seq, hidden_for_Encoder)
    Decoder = one2oneRNN(len(encoder_outputs),hidden_size,output_len,recurrent_length)
    output, _ = Decoder.forward_for_seq(encoder_outputs[-1],hidden_for_Decoder)
    return output
