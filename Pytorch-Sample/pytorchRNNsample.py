import torch
import torch.utils.data as Data
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision

# define hyper consts
EPOCH = 1
BATCH_SIZE = 64
LR = 0.003
TIME_STEP = 28
INPUT_SIZE = 28
DOWNLOAD_MINST = False
ROOT = './dataset/Minst'

# download minst dataset
train_set = torchvision.datasets.MNIST(
    root=ROOT,
    train=True, 
    transform=torchvision.transforms.ToTensor(), # transform to tensor format
    download=DOWNLOAD_MINST
)

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
test_x = test_set.test_data.type(torch.FloatTensor)[:2000]/255.
test_y = test_set.test_labels.numpy()[:2000]

class RNN(torch.nn.Module) :
    def __init__(self) :
        super(RNN, self).__init__()
        # use long short time memory rnn
        self.rnn = torch.nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1, # define how many layers in one cell
            batch_first=True # put the batch parameter at first place
        )
        self.out = torch.nn.Linear(64, 10)

    def forward(self, x) :
        rnn_out, (hid_n, hid_c) = self.rnn(x, None)
        output = self.out(rnn_out[:, -1, :]) # (batch, timestep, input)
        return output

rnn = RNN()
print(rnn) # show the structure of LSTM

# define the parameters
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR, betas=(0.9, 0.999))
loss_func = torch.nn.CrossEntropyLoss()

# train with mini-batch
for epoch in range(EPOCH) :
    for step, (batch_x, batch_y) in enumerate(train_loader) :
        curr_x = Variable(batch_x.view(-1, 28, 28))
        curr_y = Variable(batch_y)
        prediction = rnn(curr_x)
        loss = loss_func(prediction, curr_y) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0 :
            test_out = rnn(test_x)
            pred_y = torch.max(test_out, 1)[1].data.numpy()
            curr_acc = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            print("Epoch : ", epoch, ". Current accuracy is : ", curr_acc)
        
# show part of the final moudle's predictions
test_out = rnn(test_x[:20].view(-1, 28, 28))
pred_y = torch.max(test_out, 1)[1].data.numpy().squeeze()
print(pred_y, " : Final prediction\n", test_y[:20], " : Real labels")