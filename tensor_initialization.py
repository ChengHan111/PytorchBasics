import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
my_tensor = torch.tensor([[2,3,4],[4,5,6]], dtype=torch.float32,
                         device = device, requires_grad = True) # we can write requires_grad = True for reverse transform
print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)


# Other common initialization methods
x = torch.empty(size=(3,3))
x = torch.zeros((3,3))
x = torch.rand((3,3))
x = torch.ones((3,3))
x = torch.eye(5,5) #identity matrix
print(x)
x = torch.arange(start=0, end=5,step=1) #like range in python
print(x)
x = torch.linspace(start=0.1,end=1,steps=10) # ten values between
print(x)
x = torch.empty(size=(1,5)).normal_(mean=0,std=1)
x = torch.empty(size=(1,5)).uniform_(0,1)
x = torch.diag(torch.ones(3))


# HOw to initialize and convert tensors to other types(int, float, double)
tensor = torch.arange(4)
print(tensor.bool()) #0,1,2,3 convert to bool type
print(tensor.short()) #int 16
print(tensor.long()) # int 64 important
print(tensor.half()) #float 16
print(tensor.float()) #float32 important
print(tensor.double()) #float64

#Array to Tensor conversion and vice-versa
import numpy as np
np_array = np.zeros((5,5))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()

