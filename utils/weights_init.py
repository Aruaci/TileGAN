import torch.nn as nn

def weights_init(m):
    """
    Applies initial weights to certain layers following DCGAN convention.
    Checks if weight and bias exist before initialization.
    """
    classname = m.__class__.__name__
    # Initialize Convolutional layers
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        # Check if bias exists before initializing
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    # Initialize BatchNorm or InstanceNorm layers
    elif classname.find('BatchNorm') != -1 or classname.find('InstanceNorm') != -1:
        # Check if weight exists before initializing
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        # Check if bias exists before initializing
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)