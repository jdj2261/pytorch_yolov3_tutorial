import torch
import torch.nn as nn

x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])

print(torch.cat([x, y], dim=0))
#tensor([[1., 2.],
#        [3., 4.],
#        [5., 6.],
#        [7., 8.]])

print(torch.cat([x, y], dim=1))
# tensor([[1., 2., 5., 6.],
#         [3., 4., 7., 8.]])