import torch
batch_size = 10
features = 25
x = torch.rand((batch_size, features))

print(x[0].shape) #x[0,:]
print(x[:,0].shape)

print(x[2, 0:10]) #0:10 ----> [0.1.2....9]

x[0, 0] = 100

# Fancy indexing
x = torch.arange(10)
indices = [2,5,8]
print(x[indices])


x = torch.rand((3,5))
rows = torch.tensor([1,0])
cols = torch.tensor([4,0])
print(x)
print(x[rows, cols].shape)
print(x[rows, cols])
# we pick two elements 1st row 1st col and 2nd row 5th col
#  (2, 5) and (1, 1)

# More advanced indexing
x = torch.arange(10)
print(x[(x < 2) | (x > 8)]) # or
print(x[(x < 2) & (x > 8)]) # and
print(x[x.remainder(2) == 0]) # pick even element


# Useful operations
print(torch.where(x > 5, x, x*2))
# if x > 5, then stay, if not, x times 2
print(torch.tensor([0,0,1,2,2,3,4]).unique())
print(x.ndimension()) #number of dimension
print(x.numel()) #count the number of element in x

