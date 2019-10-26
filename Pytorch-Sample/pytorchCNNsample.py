import torch
import torch.utils.data as Data
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision

# define hyper consts
EPOCH = 1
BATCH_SIZE = 20
LR = 0.003
DOWNLOAD_MINST = False
ROOT = './dataset/Minst'

# download minst dataset
train_set = torchvision.datasets.MNIST(
    root=ROOT,
    train=True, 
    transform=torchvision.transforms.ToTensor(), # transform to tensor format
    download=DOWNLOAD_MINST
)

# check the dimension of the MINST dataset
print(train_set.train_data.size())

# create the loader use defined batch size
train_loader = Data.DataLoader(
    dataset=train_set, 
    batch_size=BATCH_SIZE,
    shuffle=True
)

# extract test data
test_set = torchvision.datasets.MNIST(
    root=ROOT, 
    train=False
)
test_x = Variable(torch.unsqueeze(test_set.test_data, dim=1)).type(torch.FloatTensor)[:2000]/255.
test_y = test_set.test_labels[:2000]

# build the CNN
class CNN(torch.nn.Module) :
    def __init__(self, ) :
        super(CNN, self).__init__()
        # first layer (1, 28, 28)
        self.conv1 = torch.nn.Sequential(
            # conv 卷积层 (filter)
            torch.nn.Conv2d(
                in_channels=1, # input img dimension (R,G,B) is 3, gray img is 1
                out_channels=16, # the num of the convs, each conv extract one feather
                kernel_size=5, # the size of the conv 5*5 pixels
                stride=1, # step length of convs
                padding=2
            ), # (16, 28, 28)
            # conv 
            torch.nn.ReLU(inplace=True),
            # conv pooling
            torch.nn.MaxPool2d(
                kernel_size=2 # extract the max pixel in each 2*2 pixels
            ) # (16, 14, 14)
        )
        # second layer (16, 14, 14)
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 5, 1, 2), # (32, 14, 14)
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2) # (32, 7, 7)
        )
        # the linear moudle (32, 7, 7)
        self.fc = torch.nn.Linear(32 * 7 * 7, 10)

    def forward(self, x) :
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) # (batch, 32 * 7 * 7)
        out = self.fc(x)
        return out

# define the network
myCnn = CNN()
print(myCnn)

# define the parameters
optimizer = torch.optim.Adam(myCnn.parameters(), lr=LR, betas=(0.9, 0.999))
loss_func = torch.nn.CrossEntropyLoss()

# train with mini-batch
for epoch in range(EPOCH) :
    for step, (batch_x, batch_y) in enumerate(train_loader) :
        curr_x = Variable(batch_x)
        curr_y = Variable(batch_y)
        prediction = myCnn(curr_x)
        loss = loss_func(prediction, curr_y) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    test_out = myCnn(test_x)
    pred_y = torch.max(test_out, 1)[1].data.squeeze()
    curr_acc = (pred_y == test_y).sum().item() / float(test_y.size(0))
    print("Epoch : ", epoch, ". Current accuracy is : ", curr_acc)
        
# show part of the final moudle's predictions
test_out = myCnn(test_x[:20])
pred_y = torch.max(test_out, 1)[1].data.numpy().squeeze()
print(pred_y, " : Final prediction\n", test_y[:20].numpy(), " : Real labels")