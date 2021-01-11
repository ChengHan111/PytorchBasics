import torch
x = torch.arange(9)

x_3x3 = x.view(3,3)
print(x_3x3)

x_3x3 = x.reshape(3,3)
# view and reshape both fine. Reshape Saver

y = x_3x3.t()
print(y)
# print(y.view(9)) # when we use view to transfer back, there is error and we have to use the following
print(y.contiguous().view(9))


x1 = torch.rand((2,5))
x2 = torch.rand((2,5))
#  join matrix
print(torch.cat((x1, x2), dim=0).shape)
print(torch.cat((x1, x2), dim=1).shape)


z = x1.view(-1) #magically know you want to flatten the matrix
print(z.shape)
print(z)

batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(batch, -1)
print(z.shape) # we still want to keep the batch

z = x.permute(0,2,1) # switch dimension the 0,1,2 is the order of dimension
print(z.shape)

x = torch.arange(10) #[10]
# squeeze的用法主要就是对数据的维度进行压缩或者解压。
#
# 先看torch.squeeze() 这个函数主要对数据的维度进行压缩，去掉维数为1的的维度，比如是一行或者一列这种，一个一行三列（1,3）的数去掉第一个维数为一的维度之后就变成（3）行。squeeze(a)就是将a中所有为1的维度删掉。不为1的维度没有影响。a.squeeze(N) 就是去掉a中指定的维数为一的维度。还有一种形式就是b=torch.squeeze(a，N) a中去掉指定的定的维数为一的维度。
#
# 再看torch.unsqueeze()这个函数主要是对数据维度进行扩充。给指定位置加上维数为一的维度，比如原本有个三行的数据（3），在0的位置加了一维就变成一行三列（1,3）。a.squeeze(N) 就是在a中指定位置N加上一个维数为1的维度。还有一种形式就是b=torch.squeeze(a，N) a就是在a中指定位置N加上一个维数为1的维度
print(x.unsqueeze(0).shape)
print(x.unsqueeze(1).shape)
x = torch.arange(10).unsqueeze(0).unsqueeze(1) #1x1x10
print(x.shape)
z = x.squeeze(1)
print(z.shape)