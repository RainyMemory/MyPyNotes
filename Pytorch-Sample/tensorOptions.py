from __future__ import print_function
import torch
import numpy as np

def p_func(var) :
    print("\n________________\n")
    print(var)

# Empty tensor
x_e = torch.empty(3,5)
p_func(x_e)

# Random tensor
x_r = torch.rand(5,3)
p_func(x_r)

# Zeros & Ones tensor
x_zs = torch.zeros(5,3,dtype=torch.long)
x_os = torch.ones(5,3,dtype=torch.long)
p_func(x_zs)
p_func(x_os)

# Create from current list & transform
x_cur = torch.FloatTensor(range(24))
p_func(x_cur)
x_cur = x_cur.view(3,2,4)
p_func(x_cur)

# Generate from existing tensor shape
x_cop = torch.rand_like(x_r, dtype=torch.float)
p_func(x_cop)

# size & shape
p_func(x_cur.size())
p_func(x_cur.shape)

# slice
p_func(x_r[:,2])

# single item
p_func(x_r[3,0].item())

# transfer into numpy 'narray'
z_r = x_r.numpy()
p_func(z_r)
x_r.add_(2.)
p_func(z_r)

# transfer numpy into tensor
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
p_func(a)
p_func(b)

# cut tensor & transpose(Dim<=2 | mark the two dim to transpose) & split & unbind & where
x_r = torch.narrow(input=x_r, dim=1, start=0, length=2)
p_func(x_r)
p_func(x_r.t())
p_func(torch.transpose(input=x_r, dim0=0, dim1=1))
p_func(torch.split(x_r, split_size_or_sections=3, dim=0))
p_func(torch.unbind(x_r))
p_func(torch.where(x_zs == 0, x_os, x_zs))


# More info : https://pytorch.org/docs/stable/torch.html