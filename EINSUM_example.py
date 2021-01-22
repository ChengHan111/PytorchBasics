import torch
x = torch.rand((2,3))

# Permutation of Tensors
torch.einsum('ij->ji', x)

# Summation
torch.einsum('ij->', x)

# Column sum
torch.einsum('ij->j', x)

# Row sum
torch.einsum('ij->i', x)

# Matrix-Vector Multiplication
v = torch.rand((1,3))
torch.einsum('ij, kj->ik', x, v)

# Matrix-Matrix Multiplication
torch.einsum('ij,kj->ik', x, x) #2x2: 2x3 X 3x2

# Dot product first row with first row of matrix
torch.einsum('i,i->', x[0],x[0])

# Dot product with matrix
torch.einsum('ij,ij->', x, x)

# Hadamard Product (element-wise multiplication)
# https://en.wikipedia.org/wiki/Hadamard_product_(matrices)
torch.einsum('ij,ij->ij', x, x)

# Outer product
a = torch.rand((3))
b = torch.rand((5))
torch.einsum('i,j->ij', a, b)

# Batch matrix Multiplication
a = torch.rand((3, 2, 5))
b = torch.rand((3, 5, 3))
c = torch.einsum('ijk,ikl->ijl', a, b) #we have 3 example in the batch, which we can have a output 3x2x3
# print(c.shape)
# http://christopher5106.github.io/deep/learning/2018/10/28/understand-batch-matrix-multiplication.html

# Matrix Diagonal
X = torch.rand((3,3))
torch.einsum('ii->i', x)

# Martix Trace
# zhu dui jiao xian shang ge yuan su de zong he
# https://zh.wikipedia.org/wiki/%E8%B7%A1
torch.einsum('ii->', x)

