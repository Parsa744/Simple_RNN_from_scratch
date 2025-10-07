import numpy as np
import matplotlib.pyplot as plt


def relu(x):
    """ReLU activation function using NumPy"""
    return np.maximum(0, x)


class RNN:
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
        '''print('weights initialized')
        print('wx:',self.W_x)
        print('wh:',self.W_h)
        print('wy:',self.W_y)
        print('bh:',self.bh)
        print('by:',self.by)'''
    def forward(self, input, hidden):
        self.hidden = np.dot(self.W_x, input) + np.dot(self.W_h, hidden) + self.bh
        self.hidden = relu(self.hidden)
        self.output = np.dot(self.hidden,self.W_h) + self.by
        self.output = relu(self.output)
        return self.output , self.hidden




def main():
    x1 = [1,2,2]
    H0 = [0,0,0,0]
    H0_b = [1]
    x2 = [1,2,4]
    x3 = [3,4,5]
    x4 = [5,6,7]
    myRNN = RNN(3,4,4)
    Y1,H1 = myRNN.forward(input=x1,hidden=H0)
    Y2,H2 = myRNN.forward(input=x2,hidden=H1)
    Y3,H3 = myRNN.forward(input=x3,hidden=H2)
    Y4,H4 = myRNN.forward(input=x4,hidden=H3)
    print(Y1)
    print(Y2)
    print(Y3)
    print(Y4)
    print('--------')
    print(H1)
    print(H2)
    print(H3)
    print(H4)
    ax = plt.figure().add_subplot(projection='3d')
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    ax.plot(x, y, zs=0, zdir='z', label='curve in (x, y)')


if __name__ == "__main__":
    main()