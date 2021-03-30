import torch
import torch.nn as nn
import torch.tensor as tensor

input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)
input
tensor([[[[ 1.,  2.],
          [ 3.,  4.]]]])

m = nn.UpsamplingBilinear2d(scale_factor=2)
print(m(input))

'''
tensor([[[[ 1.0000,  1.3333,  1.6667,  2.0000],
          [ 1.6667,  2.0000,  2.3333,  2.6667],
          [ 2.3333,  2.6667,  3.0000,  3.3333],
          [ 3.0000,  3.3333,  3.6667,  4.0000]]]])
'''