import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

class LinRegNet(torch.nn.Module) :
    def __init__(self, num_feather, num_embed, num_class) :
        super(LinRegNet, self).__init__()
        self.num_class = num_class
        self.num_feather = num_feather
        self.num_embed = num_feather
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(num_feather, num_embed),
            torch.nn.ReLU(),
            torch.nn.Linear(num_embed, num_class)
        )

    def forward(self, x) :
        out = self.fc(x)
        return out


def train_lin_net() :
    # create data for linear regression moudle
    x_lin_reg = torch.unsqueeze(torch.linspace(-1, 1, 200), dim=1) # expend the dim of the target tensor by add a pair of []
    y_lin_reg = x_lin_reg.pow(2) + 0.3 * torch.rand(x_lin_reg.size())
    # fit into variable
    x_lin_var = Variable(x_lin_reg)
    y_lin_var = Variable(y_lin_reg)
    # 1 feather : xPos; 20 cells; 1 output : yPos
    myNet = LinRegNet(1, 20, 1)
    # check the net framework & draw the pic
    print(myNet)
    plt.ion()
    plt.show()
    # set the optimizer & loss_function
    optimizer = torch.optim.RMSprop(myNet.parameters(), lr=0.05, alpha=0.9)
    loss_func = torch.nn.MSELoss()
    #train the moudel
    for epoch in range(30) :
        # configure prediction & loss in this iter 
        prediction = myNet(x_lin_var)
        loss = loss_func(prediction, y_lin_var)
        # active the backpropo and the optimizer
        optimizer.zero_grad()
        loss.backward()
        # refine all the parameters
        optimizer.step()
        # refresh the pic each 3 epoch
        if epoch % 3 == 0 :
            plt.cla()
            plt.scatter(x_lin_reg, y_lin_reg)
            plt.plot(x_lin_var.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
            plt.text(0.5, 0, 'LOSS : %.4f' % loss.data, fontdict={'size' : 20, 'color' : 'red'})
            plt.pause(0.5)
    plt.ioff()
    plt.show()


def train_lin_classification_net() :
    # create data for the training module
    n_data = torch.ones(100, 2)
    x0 = torch.normal(2*n_data, 1)      # class0 x data (tensor), shape=(100, 2)
    y0 = torch.zeros(100)               # class0 y data (tensor), shape=(100, 1)
    x1 = torch.normal(-2*n_data, 1)     # class1 x data (tensor), shape=(100, 2)
    y1 = torch.ones(100)                # class1 y data (tensor), shape=(100, 1)
    x_cla_ten = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
    y_cla_ten = torch.cat((y0, y1), ).type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer
    # fit into variable
    x_cla_var = Variable(x_cla_ten)
    y_cla_var = Variable(y_cla_ten)
    # init and train the net. 2 feathers : xPos,yPos; 20 cells; 2 output : probClassA,probClassB
    myNet = LinRegNet(2, 20, 2)
    print(myNet)
    plt.ion()
    plt.show()
    optimizer = torch.optim.RMSprop(myNet.parameters(), lr=0.01, alpha=0.9)
    loss_func = torch.nn.CrossEntropyLoss()
    for epoch in range(30) :
        prediction = myNet(x_cla_var)
        loss = loss_func(prediction, y_cla_var)
        # same
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # same
        if epoch % 3 == 0 :
            plt.cla()
            epo_predict = torch.max(torch.nn.functional.softmax(prediction), 1)[1]
            epo_p_y = epo_predict.data.numpy().squeeze()    
            plt.scatter(x_cla_ten[:, 0], x_cla_ten[:, 1], c=epo_p_y, s=100, lw=0, cmap='RdYlGn')
            curr_acc = sum(epo_p_y == y_cla_var.data.numpy()) / 200
            plt.text(0.5, 0, 'ACC : %.4f' % curr_acc, fontdict={'size' : 20, 'color' : 'red'})
            plt.pause(0.5)
    plt.ioff()
    plt.show()


def fast_net_construct():
    myNet = torch.nn.Sequential(
        torch.nn.Linear(2, 15),
            torch.nn.ReLU(),
            torch.nn.Linear(15, 2)
    )
    print(myNet)


if __name__ == "__main__":
    # train_lin_net()
    # train_lin_classification_net()
    print("Training done")