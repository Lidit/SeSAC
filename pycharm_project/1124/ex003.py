# nn.Conv2d - Single channel input, single channel output
import torch
import torch.nn as nn

H, W = 100, 150
input_tensor = torch.rand(size=(1, H, W))

conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
output_tensor = conv(input_tensor)

print(conv)
print("weights: ", conv.weight)
print("biases: ", conv.bias)
print("input shape: ", input_tensor.shape)
print("output shape: ", output_tensor.shape)

conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=2)
output_tensor = conv(input_tensor)

print(conv)
print("weight: ", conv.weight)
print("bias: ", conv.bias)
print("input shape: ", input_tensor.shape)
print("output shape: ", output_tensor.shape)
