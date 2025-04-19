import torch
import torch.nn as nn

# --- U-Net Generator for Conditional Inpainting ---
class UNetDown(nn.Module):
    """A downsampling block in the U-Net encoder."""
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels)) # InstanceNorm often works well in image translation
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    """An upsampling block in the U-Net decoder with skip connection."""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) # Use ReLU in decoder
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        # Concatenate skip connection feature map along channel dimension
        x = torch.cat((x, skip_input), 1)
        return x

class UNetGenerator(nn.Module):
    """
    U-Net based Generator for conditional inpainting.
    Takes masked image, mask, and condition label as input.
    Outputs the inpainted image.
    """
    def __init__(self, num_channels=3, num_classes=2, embed_size=16, ngf=64):
        super(UNetGenerator, self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.embed_size = embed_size # Size of the condition embedding

        # Input channels = image channels + mask channel + embedding channels
        input_c = num_channels + 1 + embed_size

        # --- Conditioning ---
        self.label_embedding = nn.Embedding(num_classes, embed_size)

        # --- Encoder (Downsampling) ---
        self.down1 = UNetDown(input_c, ngf, normalize=False) # 256 -> 128
        self.down2 = UNetDown(ngf, ngf * 2)                 # 128 -> 64
        self.down3 = UNetDown(ngf * 2, ngf * 4)             # 64 -> 32
        self.down4 = UNetDown(ngf * 4, ngf * 8)             # 32 -> 16
        self.down5 = UNetDown(ngf * 8, ngf * 8)             # 16 -> 8
        self.down6 = UNetDown(ngf * 8, ngf * 8)             # 8 -> 4
        self.down7 = UNetDown(ngf * 8, ngf * 8)             # 4 -> 2
        self.down8 = UNetDown(ngf * 8, ngf * 8, normalize=False) # 2 -> 1 (Bottleneck)

        # --- Decoder (Upsampling with Skip Connections) ---
        # Note: Input channels to Up block = channels from previous Up + channels from corresponding Down
        self.up1 = UNetUp(ngf * 8, ngf * 8, dropout=0.5)        # 1 -> 2
        self.up2 = UNetUp(ngf * 8 * 2, ngf * 8, dropout=0.5)    # 2 -> 4 (*2 because of skip connection)
        self.up3 = UNetUp(ngf * 8 * 2, ngf * 8, dropout=0.5)    # 4 -> 8
        self.up4 = UNetUp(ngf * 8 * 2, ngf * 8)                 # 8 -> 16
        self.up5 = UNetUp(ngf * 8 * 2, ngf * 4)                 # 16 -> 32
        self.up6 = UNetUp(ngf * 4 * 2, ngf * 2)                 # 32 -> 64
        self.up7 = UNetUp(ngf * 2 * 2, ngf)                     # 64 -> 128

        # Final layer to output image (upsample to 256)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, num_channels, kernel_size=4, stride=2, padding=1), # 128 -> 256
            nn.Tanh() # Output in [-1, 1] range
        )

    def forward(self, masked_image, mask, condition_label):
        # 1. Embed the condition label
        # Input condition_label shape: [batch_size]
        embedded_label = self.label_embedding(condition_label) # [batch_size, embed_size]

        # 2. Reshape embedding to match spatial dimensions HxW
        # Target shape: [batch_size, embed_size, H, W]
        batch_size, _, H, W = masked_image.shape
        embedded_label_map = embedded_label.view(batch_size, self.embed_size, 1, 1).expand(-1, -1, H, W)

        # 3. Concatenate inputs along channel dimension
        # Input shape: [batch_size, num_channels + 1 + embed_size, H, W]
        model_input = torch.cat([masked_image, mask, embedded_label_map], dim=1)

        # --- U-Net Pass ---
        # Encoder
        d1 = self.down1(model_input)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7) # Bottleneck

        # Decoder with Skip Connections
        u1 = self.up1(d8, d7) # Skip from down7
        u2 = self.up2(u1, d6) # Skip from down6
        u3 = self.up3(u2, d5) # Skip from down5
        u4 = self.up4(u3, d4) # Skip from down4
        u5 = self.up5(u4, d3) # Skip from down3
        u6 = self.up6(u5, d2) # Skip from down2
        u7 = self.up7(u6, d1) # Skip from down1

        # Final output layer
        generated_patch = self.final_up(u7)

        # Combine the generated patch with the unmasked input regions
        # Output is the complete generated image
        output_image = masked_image * (1.0 - mask) + generated_patch * mask

        return output_image