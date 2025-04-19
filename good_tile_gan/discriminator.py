import torch.nn as nn
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, num_channels, image_size, ndf=64):
        super(Discriminator, self).__init__()
        self.num_channels = num_channels
        self.image_size = image_size
        self.ndf = ndf

        num_intermediate_layers = int(np.log2(image_size / 8))

        layers = []
        layers.append(nn.Conv2d(num_channels, ndf, 4, 2, 1, bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        current_filters = ndf
        for i in range(num_intermediate_layers):
            next_filters = current_filters * 2
            layers.append(nn.Conv2d(current_filters, next_filters, 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(next_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            current_filters = next_filters
           
        layers.append(nn.Conv2d(current_filters, 1, kernel_size=4, stride=1, padding=0, bias=False))

        self.main = nn.Sequential(*layers)

    def forward(self, image):
        return self.main(image)