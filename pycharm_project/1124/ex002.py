# nn.Conv2d

import torch
import torch.nn as nn

H, W = 100, 150
input_tensor = torch.rand(size=(1, H, W))

conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)

print(conv.eval())

output_tensor = conv(input_tensor)

print(input_tensor.shape)
print(output_tensor.shape)


# (B, C, H, W)