import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class TileDataset(Dataset):
    def __init__(
            self,
            image_paths_dict,
            mask_base_dir,
            anomaly_types_for_training,
            image_size,
            num_channels=3,
            transform_image=None,
            transform_mask=None
            ):
        super().__init__()
        self.image_size = image_size
        self.num_channels = num_channels
        self.mask_base_dir = mask_base_dir
        self.samples = []

        self.defect_map = {name: i for i, name in enumerate(anomaly_types_for_training)}
        self.num_classes = len(self.defect_map)

        # --- Process anomaly images based on anomaly_types_for_training ---
        for category, label in self.defect_map.items():
            if category in image_paths_dict and image_paths_dict[category]:
                category_mask_dir = os.path.join(self.mask_base_dir, category)
                count = 0
                print(f"Processing category: {category}")
                for img_path in image_paths_dict[category]:
                    # Derive expected mask path from image path
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    expected_mask_name = f"{base_name}_mask.png"
                    mask_path = os.path.join(category_mask_dir, expected_mask_name)

                    if os.path.exists(mask_path):
                        self.samples.append((img_path, 'anomaly', mask_path, label))
                        count += 1
                    else:
                        print(f"  Warning: Mask not found at {mask_path} for image {img_path}")
                print(f"  -> Added {count} samples for anomaly type '{category}'.")
            else:
                 print(f"- Warning: Category '{category}' not found or empty in image paths dictionary.")

        # --- Process 'good' images ---
        # Assumes 'good' paths come from both train/good and test/good in the dict
        if 'good' in image_paths_dict and image_paths_dict['good']:
            count = 0
            print("Processing category: good")
            for img_path in image_paths_dict['good']:
                self.samples.append((img_path, 'good', None, -1))
                count += 1
            print(f"  -> Added {count} 'good' samples.")
        else:
            print("- Warning: 'good' category not found or empty in image paths dictionary.")

        if not self.samples:
            raise ValueError("No valid samples found after processing paths and masks. Check configurations.")

        print(f"-> Total {len(self.samples)} samples prepared for the dataset.")

        # Define default transformations (same as before)
        self.transform_image = transform_image or transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.transform_mask = transform_mask or transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    # --- __len__ and __getitem__ methods remain the same as the previous conditional GAN version ---
    # (The __getitem__ already handles loading the mask_path if it's not None,
    # or generating a synthetic mask if mask_path is None)

    def __len__(self):
        return len(self.samples)

    def generate_random_mask(self, size, max_shapes=5, min_size=0.05, max_size=0.3):
        """Generates a simple random mask (e.g., rectangles)."""
        mask = torch.zeros((1, size, size), dtype=torch.float32)
        num_shapes = random.randint(1, max_shapes)
        for _ in range(num_shapes):
            shape_type = random.choice(['rect'])
            if shape_type == 'rect':
                h = int(size * random.uniform(min_size, max_size))
                w = int(size * random.uniform(min_size, max_size))
                top = random.randint(0, size - h)
                left = random.randint(0, size - w)
                mask[:, top:top+h, left:left+w] = 1.0
        return mask

    def __getitem__(self, idx):
        img_path, sample_type, mask_path, condition_label = self.samples[idx]

        try:
            # --- Load PIL Images ---
            image_pil = Image.open(img_path).convert('RGB')
            mask_pil = None # Initialize mask_pil

            if sample_type == 'good':
                # For good images, we'll generate the mask later *after* potential augmentations
                # to the base image, but assign the random label now.
                target_image_pil = image_pil # Target is the good image
                condition_label = torch.randint(0, self.num_classes, (1,)).item() # Random defect class
            elif sample_type == 'anomaly':
                # Load the real mask for anomaly images
                if mask_path: # Check if mask path exists (it should for anomaly type)
                    mask_pil = Image.open(mask_path).convert('L') # Load real mask as PIL
                else:
                    # Handle case where mask path is missing for an anomaly (shouldn't happen ideally)
                    print(f"Warning: Missing mask_path for anomaly sample {idx}. Using empty mask.")
                    mask_pil = Image.new('L', image_pil.size, 0) # Create blank mask
                target_image_pil = image_pil # Target is the anomaly image
                # condition_label is already set correctly
            else:
                raise ValueError(f"Unknown sample type: {sample_type}")

            # --- Apply Synchronized Geometric Augmentations ---
            # 1. Random Horizontal Flip
            if random.random() > 0.5:
                target_image_pil = transforms.functional.hflip(target_image_pil)
                if mask_pil: # Apply same flip to mask if it exists
                    mask_pil = transforms.functional.hflip(mask_pil)

            # 2. Random Vertical Flip
            if random.random() > 0.5:
                target_image_pil = transforms.functional.vflip(target_image_pil)
                if mask_pil: # Apply same flip to mask if it exists
                    mask_pil = transforms.functional.vflip(mask_pil)

            # 3. Random Rotation (e.g., 0, 90, 180, 270 degrees for tiles)
            # Or small arbitrary rotations: angle = random.uniform(-10, 10)
            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                target_image_pil = transforms.functional.rotate(target_image_pil, angle)
                if mask_pil: # Apply same rotation to mask if it exists
                    mask_pil = transforms.functional.rotate(mask_pil, angle)

            # --- Apply Standard Transforms (Resize, ToTensor, Normalize) ---
            processed_image = self.transform_image(target_image_pil) # Transform the (potentially augmented) target image

            # --- Handle Mask Generation/Transformation ---
            if sample_type == 'good':
                # Generate synthetic mask *after* potential geometric transforms on the base image
                mask = self.generate_random_mask(self.image_size) # Generates a Tensor mask [0.0, 1.0]
            elif mask_pil:
                # Transform the (potentially augmented) real mask PIL image
                mask = self.transform_mask(mask_pil) # Transform to Tensor mask [0.0, 1.0]
            else: # Fallback empty mask if something went wrong
                mask = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)


            # Create masked input using the processed image and final mask tensor
            masked_input = processed_image * (1.0 - mask) + (-1.0 * mask)

            return masked_input, processed_image, mask, torch.tensor(condition_label).long() # Return processed_image as target

        except Exception as e:
            print(f"Error processing sample {idx} (path: {img_path}): {e}")
            # Return dummy tensors on error
            dummy_img = torch.zeros((self.num_channels, self.image_size, self.image_size))
            dummy_mask = torch.zeros((1, self.image_size, self.image_size))
            dummy_label = torch.tensor(-1).long()
            return dummy_img, dummy_img, dummy_mask, dummy_label