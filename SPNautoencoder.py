import torch.nn as nn

class SPNAutoencoder(nn.Module):
    def __init__(self):
        super(SPNAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(512, 512, kernel_size=4, stride=1, padding=0))
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(512, 512, kernel_size=4, stride=1, padding=0), dim=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.utils.spectral_norm(nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), dim=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.utils.spectral_norm(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), dim=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.utils.spectral_norm(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), dim=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.utils.spectral_norm(nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1), dim=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
