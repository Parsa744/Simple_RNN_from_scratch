from models import torchRNN
import torch
import torch.nn as nn
import torch.optim as optim





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