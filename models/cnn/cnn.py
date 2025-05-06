import torch
import torch.nn as nn

# Arjun's CNN
class CNN(nn.Module):
    def __init__(self, input_channels, version_str: str = 'v0.0.0'):
        super().__init__()
        self.version_num = version_str
        self.save_path = f'models/cnn/saves/cnn_{self.version_num}.pth'

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=2, stride=2),
            nn.ReLU(),
        )

        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, 1028),
            nn.ReLU(),
            nn.Linear(1028, 1028),
            nn.ReLU(),
            nn.Linear(1028, 128 * 32 * 32),
            nn.Unflatten(1, (128, 32, 32))
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2),  
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=8, kernel_size=2, stride=2, padding=1) # made changes to the kernel size to fit pixel cutoff issue
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
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

# Jack's CNN
class miniCNN(nn.Module):
    def __init__(self, input_channels, name: str = 'cnn', version_str: str = 'v0.0.0'):
        super().__init__()
        self.name = name
        self.latent_dim = 3 * 5 * 5
        self.version_num = version_str
        self.save_path = f'models/cnn/saves/{self.name}_{self.version_num}.pth'

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, self.latent_dim),
            nn.Sigmoid(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 32 * 32 * 32),
            nn.Unflatten(1, (32, 32, 32)),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2),  
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=2, stride=2, padding=1) # made changes to the kernel size to fit pixel cutoff issue
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    
    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

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



