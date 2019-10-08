import torch
import torch.utils.data as Data

BATCH_SIZE = 4

x = torch.linspace(1, 10, 10)       # this is x data (torch tensor)
y = torch.linspace(10, 1, 10)       # this is y data (torch tensor)

# create the dataset
my_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=my_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
    # num_workers=2 readin with more cores or threads
)

for epoch in range(5) :
    for step, (batch_x, batch_y) in enumerate(loader) :
        # simulate the training process
        print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
              batch_x.numpy(), '| batch y: ', batch_y.numpy())