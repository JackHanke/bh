import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# feed forward neural network
class FFNN(nn.Module):
    def __init__(self, input_dim: int, version_str: str = 'v0.0.0'):
        super().__init__()
        self.version_num = version_str
        self.save_path = f'models/ffnn/saves/ffnn_{self.version_num}.pth'
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x

    def save(self):
        torch.save(self.state_dict(), self.save_path)
        print(f'Saved model as {self.save_path}')

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def size_in_memory(self):
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

# AE with FFNN encoder and FFNN decoder
class AE(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        latent_dim: int,
        version_str: str = 'v0.0.0'
        ):
        super().__init__()
        self.version_num = version_str
        self.save_path = f'models/ffnn/saves/ae_{self.version_num}.pth'

        # network architecture params
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.middle_val = latent_dim
        # self.middle_val = (input_dim + latent_dim) // 2

        # encoder layers
        self.encode_layer1 = nn.Linear(self.input_dim, self.middle_val)
        self.encode_layer2 = nn.Linear(self.middle_val, self.middle_val)
        self.encode_layer3 = nn.Linear(self.middle_val, self.latent_dim)

        # decoder layers
        self.decode_layer1 = nn.Linear(self.latent_dim, self.middle_val)
        self.decode_layer2 = nn.Linear(self.middle_val, self.middle_val)
        self.decode_layer3 = nn.Linear(self.middle_val, self.input_dim)
        
    # encode x vector into latent
    def encode(self, x):
        x = self.encode_layer1(x)
        x = self.encode_layer2(x)
        x = self.encode_layer3(x)
        return x
    
    # takes x of size latent dim
    def decode(self, x):
        x = self.decode_layer1(x)
        x = self.decode_layer2(x)
        x = self.decode_layer3(x)
        return x

    # full forward pass
    def forward(self, x):
        # encode
        latent_representation = self.encode(x)
        # decode
        x = self.decode(latent_representation)
        return x

    def save(self):
        torch.save(self.state_dict(), self.save_path)
        print(f'Saved model as {self.save_path}')

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def size_in_memory(self):
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb
        