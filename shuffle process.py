import torch
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
import torch.utils.data as Data

# torch.manual_seed(1)    # reproducible

BATCH_SIZE = 2000
EPOCH = 3

# x = torch.linspace(1, 10, 10)       # this is x data (torch tensor)
# y = torch.linspace(10, 1, 10)       # this is y data (torch tensor)
x = torch.unsqueeze(torch.linspace(-math.pi, math.pi, 10000), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)
# plt.figure(num=1)
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=False,               # random shuffle for training
    num_workers=2,              # subprocesses for loading data
)


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_hidden2,  n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden2)   # hidden2 layer
        self.predict = torch.nn.Linear(n_hidden2, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = F.relu(self.hidden2(x))
        x = self.predict(x)             # linear output
        return x


net1 = Net(n_feature=1, n_hidden=10, n_hidden2=15, n_output=1)     # define the network

optimizer = torch.optim.Adam(net1.parameters(), lr=0.1)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss


def show_batch():
    for epoch in range(EPOCH):   # train entire dataset 3 times
        for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
            # train your data...
            # print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
            #       batch_x.numpy(), '| batch y: ', batch_y.numpy())

            plt.ion()  # something about plotting
            for t in range(400):

                prediction = net1(batch_x)  # input x and predict based on x
                loss = loss_func(prediction, batch_y)  # must be (1. nn output, 2. target)
                optimizer.zero_grad()  # clear gradients for next train
                loss.backward()  # back-propagation, compute gradients
                optimizer.step()  # apply gradients

                if t % 10 == 0:
                    # plot and show learning process
                    plt.cla()
                    plt.scatter(x.data.numpy(), y.data.numpy())
                    plt.plot(batch_x.data.numpy(), prediction.data.numpy(), 'r-', lw=10)
                    plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
                    plt.pause(0.1)

                plt.ioff()
                plt.show()

            plt.figure()
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.plot(batch_x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)


if __name__ == '__main__':
    show_batch()
