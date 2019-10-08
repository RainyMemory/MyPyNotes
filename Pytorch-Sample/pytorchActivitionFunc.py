import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

# create 200 figures from -5 to 5 as a 1 dim tensor
x_ten = torch.linspace(-5, 5, 200)
x_var = Variable(x_ten)
x_arr = x_ten.numpy()

# matplotlib accept numpy arrays as inputs
y_sigmoid = torch.sigmoid(x_var).data.numpy()
y_relu = torch.relu(x_var).data.numpy()
y_tanh = torch.tanh(x_var).data.numpy()
y_softplus = torch.nn.functional.softplus(x_var).data.numpy()

plt.figure("Activition Functions")
plt.subplot(221)
plt.plot(x_arr, y_sigmoid, c='red', label="Sigmoid")
plt.ylim((-1, 5))
plt.legend(loc='best')
plt.subplot(222)
plt.plot(x_arr, y_relu, c='blue', label="Relu")
plt.ylim((-1, 5))
plt.legend(loc='best')
plt.subplot(223)
plt.plot(x_arr, y_softplus, c='green', label="SoftPlus")
plt.ylim((-1, 5))
plt.legend(loc='best')
plt.subplot(224)
plt.plot(x_arr, y_tanh, c='yellow', label="tanh")
plt.ylim((-1, 5))
plt.legend(loc='best')
plt.show()