import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

# torch.manual_seed(1)    # reproducible

x = torch.unsqueeze(torch.linspace(-math.pi, math.pi, 200), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)
# train_data = x.sin() + 0.2*torch.rand(x.size())


# torch can only train on Variable, so convert them to Variable
# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
# x, y = Variable(x), Variable(y)

plt.figure(num=1)
plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()

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
print(net1)  # net architecture


# """
# 快速搭建网络
# """
# net1 = torch.nn.Sequential(
#     torch.nn.Linear(1, 10),
#     torch.nn.ReLU(),
#     torch.nn.Linear(10, 15),
#     torch.nn.ReLU(),
#     torch.nn.Linear(15, 1)
# )
# print(net1)

optimizer = torch.optim.Adam(net1.parameters(), lr=0.1)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

plt.ion()   # something about plotting

for t in range(500):
    prediction = net1(x)  # input x and predict based on x

    loss = loss_func(prediction, y)  # must be (1. nn output, 2. target)

    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients

    if t % 10 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

    plt.ioff()
    plt.show()

plt.figure()
plt.subplot(3, 1, 1)
plt.title('Net1')
plt.scatter(x.data.numpy(), y.data.numpy())
plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)


# 保存神经网络
torch.save(net1, 'net.pkl')
torch.save(net1.state_dict(), 'net_params.pkl')

# 将网络1提取到网络2
net2 = torch.load('net.pkl')
prediction = net2(x)

# plot result
plt.subplot(3, 1, 2)
plt.title('Net2')
plt.scatter(x.data.numpy(), y.data.numpy())
plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
# plt.show()

# 将网络1的参数赋值给网络3
# net3 = torch.nn.Sequential(
#     torch.nn.Linear(1, 10),
#     torch.nn.ReLU(),
#     torch.nn.Linear(10, 15),
#     torch.nn.ReLU(),
#     torch.nn.Linear(15, 1)
# )

net3 = Net(n_feature=1, n_hidden=10, n_hidden2=15, n_output=1)
net3.load_state_dict(torch.load('net_params.pkl'))
prediction = net3(x)
print(net3)

plt.subplot(3, 1, 3)
plt.title('Net3')
plt.scatter(x.data.numpy(), y.data.numpy())
plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
plt.show()
