import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    def __init__(self, latent_dim, num_channels, image_size, ngf=64):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        self.image_size = image_size
        self.ngf = ngf

        
        num_intermediate_layers = int(np.log2(image_size / 8))

        layers = []
        
        initial_filters = ngf * (2**num_intermediate_layers)
        layers.append(nn.ConvTranspose2d(latent_dim, initial_filters, 4, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(initial_filters))
        layers.append(nn.ReLU(True))

        current_filters = initial_filters
        for i in range(num_intermediate_layers):
            next_filters = current_filters // 2
            layers.append(nn.ConvTranspose2d(current_filters, next_filters, 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(next_filters))
            layers.append(nn.ReLU(True))
            current_filters = next_filters

        layers.append(nn.ConvTranspose2d(current_filters, num_channels, 4, 2, 1, bias=False))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, noise):
        return self.main(noise)