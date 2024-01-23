import torch.nn as nn
import torch

input_tensor = torch.tensor(
    [[0.3471, 0.4547, -0.2356, 0.0928, 0.3471, 0.4547, -0.2356, 0.0928, 0.3471, 0.4547 ]]
)

print(input_tensor)

linear_layer = nn.Linear(in_features=10, out_features=5)

# Create network with three linear layers

model = nn.Sequential(
    nn.Linear(10, 18),
    nn.Linear(18, 20),
    nn.Linear(20, 5)
)

output = linear_layer(input_tensor)
print(output)


