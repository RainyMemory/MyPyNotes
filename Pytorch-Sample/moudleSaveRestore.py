import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)
x, y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)
plt.figure(1, figsize=(10, 3))

def save_net():
    myNet = torch.nn.Sequential(
        torch.nn.Linear(1, 15),
            torch.nn.ReLU(),
            torch.nn.Linear(15, 1)
    )
    optimizer = torch.optim.RMSprop(myNet.parameters(), lr=0.03, alpha=0.9)
    loss_func = torch.nn.MSELoss()
    for epoch in range(30) :
        prediction = myNet(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    plot_curr_net(131, prediction)
    # save the current net moudle
    torch.save(myNet, './Pytorch-Sample/net1.pkl') # save the entire net
    torch.save(myNet.state_dict(), './Pytorch-Sample/net1_params.pkl') # save the current parameters in the network


def restore_net() :
    myNet = torch.load('./Pytorch-Sample/net1.pkl') # load the whole net work
    prediction = myNet(x)
    plot_curr_net(132, prediction)


def restore_net_param() :
    myNet = torch.nn.Sequential(
        torch.nn.Linear(1, 15),
            torch.nn.ReLU(),
            torch.nn.Linear(15, 1)
    )
    myNet.load_state_dict(torch.load('./Pytorch-Sample/net1_params.pkl')) # load the params, will be more efficient
    prediction = myNet(x)
    plot_curr_net(133, prediction)


def plot_curr_net(position, prediction) :
    plt.subplot(position)
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)


if __name__ == "__main__":
    save_net()
    restore_net()
    restore_net_param()
    plt.show()