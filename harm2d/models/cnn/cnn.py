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
            buffer_size += buffer.nelement() *  buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

# the 3D CNN for full 3 dimensional bh emulation
#Jack's 3dcnn
class JACK_CNN_3D(nn.Module):
    def __init__(self, input_channels: int = 8, name: str = '3dcnn', version_str: str = 'v0.0.0'):
        super().__init__()
        # model information and metadata
        self.name = name
        self.version_num = version_str
        self.save_path = f'models/cnn/saves/{self.name}_{self.version_num}.pth'
        # middle - most latent dimension
        self.latent_dim = 3 * 5 * 5
        
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


        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 56 * 12 * 24, self.latent_dim),
            nn.Unflatten(1, (4, 56, 12, 24)),
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

    @torch.no_grad()
    def inference(self, x):
        return self.forward(x)
    
    def save(self, save_path:str = None):
        if save_path is None:
            torch.save(self.state_dict(), self.save_path)
            print(f'Saved model as {self.save_path}')
        else:
            torch.save(self.state_dict(), save_path)
            print(f'Saved model as {save_path}')

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

##########################################################################################
# Arjun's 3D CNN
class CNN_DEPTH(nn.Module):
    def __init__(self, input_channels: int = 8, name: str = "b3", version_str: str = 'v0.0.0'):
        super().__init__()

        # model information and metadata
        self.name = name
        self.version_num = version_str
        self.save_path = f'models/cnn/saves/{self.name}_{self.version_num}.pth'
        
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, groups=input_channels),
            nn.Conv3d(in_channels=input_channels, out_channels=32, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, groups=32),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride= 2, padding=1, groups=64),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride= 2, padding=1, groups=64),
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride= 2, padding=1, groups=64),
            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, stride= 2, padding=1, groups=64),
            nn.Conv3d(in_channels=512, out_channels=1024, kernel_size=1),
            nn.GELU(),
            )

        self.bottleneck_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 14 * 3 * 6, 512),
            nn.GELU(),
            nn.Linear(512, 1024 * 14 * 3 * 6),
            nn.Unflatten(1, (1024, 14, 3, 6)),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose3d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv3d(in_channels=64, out_channels=8, kernel_size=1)
            )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck_layers(x)
        x = self.decoder(x)
        return x
        
    # encode raw x
    def encode(self, x):
        return self.encoder(x)

    # decode latent x
    def decode(self, x):
        return self.decoder(x)

    # bottleneck layer
    def bottleneck(self, x):
        return self.bottleneck_layers(x)
    
    @torch.no_grad()
    def inference(self, x):
        return self.forward(x)
    
    def save(self, save_path:str = None):
        if save_path is None:
            torch.save(self.state_dict(), self.save_path)
            print(f'Saved model as {self.save_path}')
        else:
            torch.save(self.state_dict(), save_path)
            print(f'Saved model as {save_path}')

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

##########################################################################################

class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock3D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(True),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(in_channels)
        )
        self.relu = nn.ReLU(True)

    def forward(self, x):
        residual = x
        x = self.block(x)
        x += residual
        return self.relu(x)


class CNN_3D(nn.Module):
    def __init__(self, input_channels: int = 8, name: str = '3dcnn', version_str: str = 'v0.0.2'):
        super().__init__()
        self.name = name
        self.version_num = version_str
        self.save_path = f'models/cnn/saves/{self.name}_{self.version_num}.pth'
        self.latent_dim = 1024

        self.encoder = nn.Sequential(
            nn.Conv3d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            ResidualBlock3D(16),

            #block 2
            nn.MaxPool3d((2, 2, 2)), 
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            ResidualBlock3D(32),

            #block 3
            nn.MaxPool3d((2, 2, 2)), 
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            ResidualBlock3D(64),

            # block 4
            nn.MaxPool3d((2, 2, 2)),  
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            ResidualBlock3D(128),

            # block 5
            nn.MaxPool3d((2, 2, 2)), 
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            ResidualBlock3D(256),

            # block 6
            nn.MaxPool3d((2, 1, 2)), 
            nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            ResidualBlock3D(512),

            # final layers
            nn.AdaptiveAvgPool3d((7, 3, 3)),  
            nn.Flatten(), 
            nn.Linear(512 * 7 * 3 * 3, self.latent_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 512 * 7 * 3 * 3),
            nn.Unflatten(1, (512, 7, 3, 3)),
            nn.ReLU(),

            # block 1
            ResidualBlock3D(512),
            nn.ConvTranspose3d(512, 256, kernel_size=(4,4,4), stride=(2,1,2), padding=1),  
            nn.BatchNorm3d(256),
            nn.ReLU(),

            # block 2
            ResidualBlock3D(256),
            nn.ConvTranspose3d(256, 128, kernel_size=(4,4,4), stride=2, padding=1), 
            nn.BatchNorm3d(128),
            nn.ReLU(),

            # block 3
            ResidualBlock3D(128),
            nn.ConvTranspose3d(128, 64, kernel_size=(4,4,4), stride=2, padding=1), 
            nn.BatchNorm3d(64),
            nn.ReLU(),

            # block 4
            ResidualBlock3D(64),
            nn.ConvTranspose3d(64, 32, kernel_size=(4,4,4), stride=2, padding=1), 
            nn.BatchNorm3d(32),
            nn.ReLU(),

            # block 5
            ResidualBlock3D(32),
            nn.ConvTranspose3d(32, 16, kernel_size=(4,4,4), stride=2, padding=1), 
            nn.BatchNorm3d(16),
            nn.ReLU(),

            # block 6
            ResidualBlock3D(16),
            nn.ConvTranspose3d(16, 8, kernel_size=1, stride=1, padding=0), 
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    @torch.no_grad()
    def inference(self, x):
        return self.forward(x)

    def save(self, save_path=None):
        if save_path is None:
            save_path = self.save_path
        torch.save(self.state_dict(), save_path)
        print(f'Saved model as {save_path}')

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


class CNN_3D_UnetStyle(nn.Module):
    def __init__(self, input_channels: int = 8, name: str = '3dcnn_unet', version_str: str = 'v0.0.3'):
        super().__init__()
        self.name = name
        self.version_num = version_str
        self.save_path = f'models/cnn/saves/{self.name}_{self.version_num}.pth'
        self.latent_dim = 1024
        self.enc_block1 = nn.Sequential(
            nn.Conv3d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            ResidualBlock3D(16)
        )

        self.enc_block2 = nn.Sequential(
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            ResidualBlock3D(32)
        )

        self.enc_block3 = nn.Sequential(
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            ResidualBlock3D(64)
        )

        self.enc_block4 = nn.Sequential(
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            ResidualBlock3D(128)
        )

        self.enc_block5 = nn.Sequential(
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            ResidualBlock3D(256)
        )

        self.enc_block6 = nn.Sequential(
            nn.MaxPool3d((2, 1, 2)),
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            ResidualBlock3D(512)
        )

        #latentspace
        self.latent = nn.Sequential(
            nn.AdaptiveAvgPool3d((7, 3, 3)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 3 * 3, self.latent_dim),
            nn.ReLU()
        )
        self.dec_input_proj = nn.Sequential(
            nn.Linear(self.latent_dim, 512 * 7 * 3 * 3),
            nn.Unflatten(1, (512, 7, 3, 3)),
            nn.ReLU()
        )

        self.dec_block1 = nn.Sequential(
            ResidualBlock3D(512),
            nn.ConvTranspose3d(512, 256, kernel_size=(4,4,4), stride=(2,1,2), padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU()
        )

        self.dec_block2 = nn.Sequential(
            ResidualBlock3D(256 * 2),  # skip connection from enc block5
            nn.ConvTranspose3d(256 * 2, 128, kernel_size=(4,4,4), stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )

        self.dec_block3 = nn.Sequential(
            ResidualBlock3D(128 * 2),  # skip connection from enc block4
            nn.ConvTranspose3d(128 * 2, 64, kernel_size=(4,4,4), stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )

        self.dec_block4 = nn.Sequential(
            ResidualBlock3D(64 * 2),  # skip connection from enc block3
            nn.ConvTranspose3d(64 * 2, 32, kernel_size=(4,4,4), stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )

        self.dec_block5 = nn.Sequential(
            ResidualBlock3D(32 * 2),  # skip connection from enc block2
            nn.ConvTranspose3d(32 * 2, 16, kernel_size=(4,4,4), stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU()
        )

        self.dec_block6 = nn.Sequential(
            ResidualBlock3D(16 * 2),  # skip connection from enc block1
            nn.ConvTranspose3d(16 * 2, 8, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder path
        e1 = self.enc_block1(x)     
        e2 = self.enc_block2(e1)    
        e3 = self.enc_block3(e2)    
        e4 = self.enc_block4(e3)    
        e5 = self.enc_block5(e4)    
        e6 = self.enc_block6(e5)   

        # bottleneck
        z = self.latent(e6)

        # decoder input
        d_in = self.dec_input_proj(z)  

        # decoder path w skip connections
        d1 = self.dec_block1(d_in)                      
        d1 = torch.cat([d1, e5], dim=1)                 

        d2 = self.dec_block2(d1)                        
        d2 = torch.cat([d2, e4], dim=1)                 

        d3 = self.dec_block3(d2)                        
        d3 = torch.cat([d3, e3], dim=1)                 

        d4 = self.dec_block4(d3)                        
        d4 = torch.cat([d4, e2], dim=1)                 

        d5 = self.dec_block5(d4)                        
        d5 = torch.cat([d5, e1], dim=1)                 

        out = self.dec_block6(d5)                       

        return out

    def encode(self, x):
        e1 = self.enc_block1(x)
        e2 = self.enc_block2(e1)
        e3 = self.enc_block3(e2)
        e4 = self.enc_block4(e3)
        e5 = self.enc_block5(e4)
        e6 = self.enc_block6(e5)
        z = self.latent(e6)
        return z

    def decode(self, z):
        d_in = self.dec_input_proj(z)
        d1 = self.dec_block1(d_in)
        d2 = self.dec_block2(torch.cat([d1, e5], dim=1))
        d3 = self.dec_block3(torch.cat([d2, e4], dim=1))
        d4 = self.dec_block4(torch.cat([d3, e3], dim=1))
        d5 = self.dec_block5(torch.cat([d4, e2], dim=1))
        out = self.dec_block6(torch.cat([d5, e1], dim=1))
        return out

    @torch.no_grad()
    def inference(self, x):
        return self.forward(x)

    def save(self, save_path=None):
        if save_path is None:
            save_path = self.save_path
        torch.save(self.state_dict(), save_path)
        print(f'Saved model as {save_path}')

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


if __name__ == '__main__':
    import os

    # testing for sc stuff
    sc_testing = True
    if sc_testing:
        model = CNN_3D()
        # to run the unet model:
        # model = CNN_3D_UnetStyle() 
        save_path = os.environ['HOME']+'/bh/data.pkl'
        data = torch.load(f=save_path)
        print("Input shape:", data.shape)

        pred = model.forward(data)
        print("Output shape:", pred.shape)

        print(f'Params: {model.num_params()}')
        print(f'Mem size: {model.size_in_memory():.2f} MB')