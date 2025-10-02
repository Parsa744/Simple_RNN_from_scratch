import numpy as np

def relu(x):
	return max(0.0, x)

class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
    def init_weights(self):
        self.W_x = np.random.randn(self.hidden_size, self.input_size) * 0.01
        self.W_h = np.random.randn(self.hidden_size, self.hidden_size) * 0.01
        self.W_y = np.random.randn(self.output_size, self.hidden_size)
        self.bh = np.random.randn(self.hidden_size) * 0.01
        self.by = np.random.randn(self.output_size) * 0.01
    def forward(self, input, hidden):
        self.hidden = np.dot(self.W_xh, input) + np.dot(self.W_hh, hidden) + self.bh
        self.hidden = relu(self.hidden)
        self.output = np.dot(self.W_hy, self.hidden) + self.by
        self.output = np.exp(self.output) / np.sum(np.exp(self.output))
        return self.output
