import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator for conditional image evaluation.
    Takes image and condition label as input.
    Outputs a map of predictions (patches).
    """
    def __init__(self, num_channels=3, num_classes=2, embed_size=16, ndf=64):
        super(PatchGANDiscriminator, self).__init__()
        self.num_classes = num_classes
        self.embed_size = embed_size

        # Input channels = image channels + embedding channels
        input_c = num_channels + embed_size

        # --- Conditioning ---
        self.label_embedding = nn.Embedding(num_classes, embed_size)

        # --- Layers ---
        # Output patch size depends on layers, strides, padding. E.g., 30x30 for 256x256 input
        self.model = nn.Sequential(
            # Input: [batch, input_c, 256, 256]
            nn.Conv2d(input_c, ndf, kernel_size=4, stride=2, padding=1), # No norm on first layer
            nn.LeakyReLU(0.2, inplace=True),
            # Shape: [batch, ndf, 128, 128]

            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Shape: [batch, ndf*2, 64, 64]

            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Shape: [batch, ndf*4, 32, 32]

            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1, bias=False), # Stride 1 here
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Shape: [batch, ndf*8, 31, 31] (adjust padding if needed for exact size)

            # Final layer: Output a 1-channel prediction map
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)
            # Shape: [batch, 1, 30, 30] (adjust padding for exact size)
            # No Sigmoid here if using BCEWithLogitsLoss or WGAN loss
        )

    def forward(self, image, condition_label):
        # 1. Embed the condition label
        embedded_label = self.label_embedding(condition_label) # [batch_size, embed_size]

        # 2. Reshape embedding to match spatial dimensions HxW
        batch_size, _, H, W = image.shape
        embedded_label_map = embedded_label.view(batch_size, self.embed_size, 1, 1).expand(-1, -1, H, W)

        # 3. Concatenate inputs along channel dimension
        # Input shape: [batch_size, num_channels + embed_size, H, W]
        model_input = torch.cat([image, embedded_label_map], dim=1)

        # 4. Pass through convolutional layers
        return self.model(model_input)