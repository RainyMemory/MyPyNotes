import torch
from torch.autograd import Variable

my_tensor = torch.FloatTensor([[1, 2], [3, 5]])
# requires_grad=True : the gradient descent of the tensor will be caculated in the loss function/backpropogation
my_variable = Variable(my_tensor, requires_grad=True)
print("Variable:\n", my_variable)

ten_out = torch.mean(my_tensor * my_tensor)
var_out = torch.mean(my_variable * my_variable)
var_out.backward()
# check the gradient
print(my_variable.grad)
print(my_variable.data)

