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

# mini CNN for testing 
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
        
    # forward pass on x
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    # encode raw x
    def encode(self, x):
        return self.encoder(x)

    # decode latent x
    def decode(self, x):
        return self.decoder(x)

    # inference on x
    @torch.no_grad()
    def inference(self, x):
        return self.forward(x)

    # save model to self.save_path
    def save(self):
        torch.save(self.state_dict(), self.save_path)
        print(f'Saved model as {self.save_path}')

    # return number of parameters
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    # return size of model in memory
    def size_in_memory(self):
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

# the 3D CNN for full 3 dimensional bh emulation
class CNN_3D(nn.Module):
    def __init__(self, input_channels: int = 8, name: str = '3dcnn', version_str: str = 'v0.0.0'):
        super().__init__()
        # model information and metadata
        self.name = name
        self.version_num = version_str
        self.save_path = f'harm2d/models/cnn/saves/{self.name}_{self.version_num}.pth'
        # middle - most latent dimension
        self.latent_dim = 3 * 5 * 5

        # total data is shape (batch_size, 224, 48, 96)
        
        # encoder stack
        self.encoder = nn.Sequential(
            nn.Conv3d(
                in_channels=input_channels, 
                out_channels=4, 
                kernel_size=(3,3,3), 
                stride=1, 
                padding=2
            ),
            nn.MaxPool3d((4, 4, 4), stride=(2, 2, 2)),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=4, 
                out_channels=4, 
                kernel_size=(3,3,3), 
                stride=1,
                padding=2
            ),
            nn.MaxPool3d((4, 4, 4), stride=(2, 2, 2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4*224*48*96 // (4*4*4), self.latent_dim),
            nn.Sigmoid(),
        )

        # decoder stack
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 4*224*48*96 // (4*4*4)),
            nn.Unflatten(1, (4, (224//4), (48//4), (96//4))),
            nn.ConvTranspose3d(
                in_channels=4, 
                out_channels=4, 
                kernel_size=(4,4,4), 
                stride=2, 
                padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose3d(
                in_channels=4, 
                out_channels=8, 
                kernel_size=(4,4,4), 
                stride=2,
                padding=1
            ),
        )

    # forward pass on x
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    # encode raw x
    def encode(self, x):
        return self.encoder(x)

    # decode latent x
    def decode(self, x):
        return self.decoder(x)

    # inference on x
    @torch.no_grad()
    def inference(self, x):
        return self.forward(x)
    
    # save model to self.save_path
    def save(self, save_path:str = None):
        if save_path is None:
            torch.save(self.state_dict(), self.save_path)
            print(f'Saved model as {self.save_path}')
        else:
            torch.save(self.state_dict(), save_path)
            print(f'Saved model as {save_path}')

    # return number of parameters
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    # return size of model in memory
    def size_in_memory(self):
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb


if __name__ == '__main__':
    # testing for sc stuff
    sc_testing = True
    if sc_testing:
        model = CNN_3D()
        # read in saved data for fast latency
        save_path = os.environ['HOME']+'/bh/data.pkl'
        data = torch.load(f=save_path)
        print(data.shape)
        
        pred = model.forward(data)
        print(f'pred shape: {pred.shape}')
        
        print(f'Params: {model.num_params()}')
        print(f'Mem size: {model.size_in_memory()}')
