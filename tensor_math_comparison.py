import torch
x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7])

# Addition
z1 = torch.empty(3)
torch.add(x,y,out=z1)

z2 = torch.add(x,y)
z = x + y

# Subtraction
z = x - y

# Division
z = torch.true_divide(x,y) #1/9 2/8 3/7 separately

# inplace operations
t = torch.zeros(3)
t.add_(x)
# when we have '_', it means that it is an inplace operation, where we do not use a new space for memo
t += x # t = t + x

# Exponentiation
z = x.pow(2)
print(z)

z = x ** 2
# Simple comparison
z = x > 0
print(z)

# Matrix Multiplication
x1 = torch.rand((2,5))
x2 = torch.rand((5,3))
x3 = torch.mm(x1,x2) #2x3
x3 = x1.mm(x2)

# Matrix exponentiation
matrix_exp = torch.rand(5,5)
print(matrix_exp.matrix_power(3))

# elemnt wise mult
# yuan su xiang cheng
z = x * y
print(z)

# dot product
z = torch.dot(x,y)
print(z)

# Batch Matrix Multiplication >2D
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1,tensor2) # (batch, n , p)


# Example of Broadcasting
x1 = torch.rand((5,5))
x2 = torch.rand((1,5))

z = x1 - x2
# it doesnt make sense but in numpy and torch it is. x2 is expanded to 5x5 and each row are the same
#automatically expanded

z = x1 ** x2


# Other useful tensor operations
sum_x = torch.sum(x,dim=0)
values, indices = torch.max(x, dim=0) # x.max(dim = 0)
values, indices = torch.min(x, dim=0)
abs_x = torch.abs(x)
z = torch.argmax(x, dim=0) # special case for max, since it only return the position of the max value
z = torch.argmin(x, dim=0)
mean_x = torch.mean(x.float(), dim=0)
z = torch.eq(x,y) #check equal return True or False
print(z)
sorted_y,indices = torch.sort(y, dim=0, descending=False)  #increasing order

z = torch.clamp(x, min=0, max=10) # check the values in x which is less than zero and set them to zero, and larger than 10 set to 10

# Some logic
x = torch.tensor([1,0,1,1,1], dtype=torch.bool)
z = torch.any(x)
z = torch.all(x) # all should be one then True, else False
