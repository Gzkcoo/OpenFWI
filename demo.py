import torch
l1loss = torch.nn.L1Loss()
data = torch.tensor([[1.],[7.],[3.]])
lable =torch.tensor([[1.], [1.], [1.]])
loss = l1loss(data, lable)
print(loss)