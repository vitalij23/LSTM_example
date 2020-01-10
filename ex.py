# from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt


class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.hidden_size = 51
        self.input_size = 1
        self.levels = 2
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.levels)
        self.linear = nn.Linear(self.hidden_size, 1)

    def forward(self, input, future=0):  # 97, 999
        outputs = []
        self.hidden = (
            torch.rand(self.levels, input.size(0), self.hidden_size, dtype=torch.double),  # layers, batch, hidden
            torch.rand(self.levels, input.size(0), self.hidden_size, dtype=torch.double))
        if torch.cuda.is_available():
            self.hidden = (self.hidden[0].cuda(), self.hidden[1].cuda())

        # parallel in mini-batch
        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):  # [97, 1]
            input_t = input_t.view(1, -1, 1)  # [1, 97, 1]
            h_t, self.hidden = self.lstm(input_t, self.hidden)
            output = self.linear(h_t)  # [1, 97, 1]
            outputs += [output]  # 999 list of [1, 97, 1]
        for i in range(future):  # if we should predict the future
            h_t, self.hidden = self.lstm(output, self.hidden)
            output = self.linear(h_t)
            outputs += [output]
        outputs = torch.stack(outputs, -1).squeeze()  # list of [1, 97, 1] -> [1, 97, 1, 999] -> [97, 999]
        return outputs


def main():
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)

    # load data and make training set
    data: np.array = torch.load('traindata.pt')
    print("batches:", len(data), len(data[0]), type(data), data.shape)
    # [100, 1000] we use 97 inputes for train and 3 for test [97, 999]
    # 100 batches - we learn one function at all batches
    sl = 97  # train amount
    input = torch.from_numpy(data[:sl, :-1])  # range (-1 1) first sl, without last 1
    # [100, 1000] we use 97 inputes for train and 3 for test [97, 999]
    target = torch.from_numpy(data[:sl, 1:]).squeeze()  # without first 1
    print("target", target.size())
    test_input = torch.from_numpy(data[sl:, :-1])  # last sl, without last
    test_target = torch.from_numpy(data[sl:, 1:])  # last sl, without first
    # GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        input = input.cuda()  # GPU
        target = target.cuda()  # GPU
        test_input = test_input.cuda()
        test_target = test_target.cuda()

    # build the model
    seq = Sequence()
    seq.double()
    seq = seq.to(device)  # GPU
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.7)
    # begin to train
    for i in range(10):
        print('STEP: ', i)

        def closure():
            optimizer.zero_grad()
            out = seq(input)  # forward - reset state
            loss = criterion(out, target)
            print('loss:', loss.item())
            loss.backward()
            return loss

        optimizer.step(closure)
        print('begin to predict, no need to track gradient here')
        with torch.no_grad():
            future = 1000
            pred = seq(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            print('test loss:', loss.item())
            # GPU
            if torch.cuda.is_available():
                pred = pred.cpu()
            y = pred.detach().numpy()
        # draw the result
        plt.figure(figsize=(30, 10))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        def draw(yi, color):
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth=2.0)
            plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth=2.0)

        draw(y[0], 'r')
        draw(y[1], 'g')
        # draw(y[2], 'b')
        # draw(y[3], 'b')
        plt.savefig('predict%d.pdf' % i)
        plt.close()


if __name__ == '__main__':
   main()
