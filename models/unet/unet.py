import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, input_channels=8, version_str: str = 'v0.0.0'):
        super().__init__()
        self.version_num = version_str
        self.save_path = f'models/unet/saves/unet_{self.version_num}.pth'

        # Encoder path
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 32, kernel_size=2, stride=2),
            nn.ReLU()
        )

        # Final output layer
        self.output = nn.Conv2d(64, 8, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)

        # Bottleneck
        x3 = self.bottleneck(x3)

        # Decoder
        d1 = self.decoder1(x3)
        d1 = torch.cat([d1, x2], dim=1)

        d2 = self.decoder2(d1)
        d2 = torch.cat([d2, x1], dim=1)

        out = self.output(d2)
        return out

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