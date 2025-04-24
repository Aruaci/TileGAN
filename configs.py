ANOMALY_TYPES_TO_TRAIN = ["crack", "glue_strip", "gray_stroke", "oil", "rough"]

IMG_SIZE = 256 
NUM_CHANNELS = 3     # Image channels (RGB)
NUM_CLASSES = len(ANOMALY_TYPES_TO_TRAIN) # Number of defect conditions
EMBED_SIZE = 16      # Size of the condition embedding vector (in G and D)
NGF = 64             # Base number of features for Generator
NDF = 64             # Base number of features for Discriminator
BATCH_SIZE = 32      # Adjust based on VRAM
EPOCHS = 150         # Number of training epochs
LR_G = 0.0002        # Learning rate for Generator
LR_D = 0.0002        # Learning rate for Discriminator
BETA1 = 0.5          # Adam optimizer beta1
BETA2 = 0.999        # Adam optimizer beta2
LAMBDA_L1 = 50.0