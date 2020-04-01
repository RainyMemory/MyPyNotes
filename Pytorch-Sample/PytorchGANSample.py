import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Hyper Parameters
BATCH_SIZE = 64
LR_G = 0.0001           # learning rate for generator
LR_D = 0.0001           # learning rate for discriminator

N_IDEAS = 5             # think of this as number of ideas for generating an art work (Generator)
ART_COMPONENTS = 15     # it could be total point G can draw in the canvas
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])

def artist_work():      # generate a list of pictures that marked as the origin art works
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    paintings = a * np.power(PAINT_POINTS, 2) + (a - 1)
    paintings = torch.from_numpy(paintings).float()
    return paintings

Gen = nn.Sequential(    # try to generate a picture
    nn.Linear(N_IDEAS, 128),
    nn.ReLU(),
    nn.Linear(128, ART_COMPONENTS)
)

# the discriminator need to have a reverse structure compared with the generator
Dis = nn.Sequential(    # try to identify the generated picture 
    nn.Linear(ART_COMPONENTS, 128),
    nn.ReLU(),
    nn.Linear(128, 4),
    nn.ReLU(),
    nn.Linear(4,1),
    nn.Sigmoid()
)

Optim_Gen = torch.optim.Adam(Gen.parameters(), lr=LR_G)
Optim_Dis = torch.optim.Adam(Dis.parameters(), lr=LR_D)

plt.ion()   # something about continuous plotting

for step in range(2000):
    artist_paitings = artist_work()     # get the pictures
    
    G_ideas = torch.randn(BATCH_SIZE, N_IDEAS)
    G_paintings = Gen(G_ideas)

    D_probOrin = Dis(artist_paitings)
    D_probGen = Dis(G_paintings)

    D_loss = - torch.mean(torch.log(D_probOrin) + torch.log(1. - D_probGen))
    G_loss = torch.mean(torch.log(1. - D_probGen))
    Optim_Dis.zero_grad()
    D_loss.backward(retain_graph = True)         # reusing computational graph, keep them for the next brakprop
    Optim_Dis.step()
    Optim_Gen.zero_grad()
    G_loss.backward()
    Optim_Gen.step()

    if step % 50 == 0:  # plotting
        plt.cla()
        plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting',)
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
        plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % D_probOrin.data.numpy().mean(), fontdict={'size': 13})
        plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 13})
        plt.ylim((0, 3));plt.legend(loc='upper right', fontsize=10);plt.draw();plt.pause(0.01)

plt.ioff()
plt.show()