import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import glob
from torchvision import transforms
import math

def plot_mask_heatmaps(categories_to_plot, image_size, mask_base_dir, test_image_dir):

    num_categories = len(categories_to_plot)
    if num_categories == 0:
        print("No categories specified for plotting.")
        return

    cols = 2 if num_categories > 3 else num_categories
    rows = math.ceil(num_categories / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), squeeze=False)
    axes = axes.flatten()

    mask_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
    ])

    plot_index = 0
    for category_to_plot in categories_to_plot:
        if plot_index >= len(axes):
            break

        ax = axes[plot_index]

        category_mask_dir = os.path.join(mask_base_dir, category_to_plot)
        category_image_dir = os.path.join(test_image_dir, category_to_plot)

        mask_paths_to_plot = []
        print(f"\nSearching for masks for category '{category_to_plot}'...")

        image_files = glob.glob(os.path.join(category_image_dir, '*.png'))
        if not image_files:
            print(f"  Info: No images found in {category_image_dir}. Skipping category '{category_to_plot}'.")
            ax.set_title(f"'{category_to_plot}' (No Images Found)")
            ax.axis("off")
            plot_index += 1
            continue

        for img_path in image_files:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            expected_mask_name = f"{base_name}_mask.png"
            mask_path = os.path.join(category_mask_dir, expected_mask_name)
            if os.path.exists(mask_path):
                mask_paths_to_plot.append(mask_path)

        num_masks = len(mask_paths_to_plot)
        if num_masks == 0:
            print(f"  Info: No mask files found for category '{category_to_plot}' in {category_mask_dir}")
            ax.set_title(f"'{category_to_plot}' (No Masks Found)")
            ax.axis("off")
            plot_index += 1
            continue

        print(f"  Found {num_masks} masks for '{category_to_plot}'. Accumulating...")
        mask_accumulator = np.zeros((image_size, image_size), dtype=np.float32)

        for i, mask_path in enumerate(mask_paths_to_plot):
            try:
                mask_pil = Image.open(mask_path).convert('L')
                mask_resized_pil = mask_transform(mask_pil)
                mask_np = np.array(mask_resized_pil, dtype=np.float32)
                mask_np = mask_np / 255.0 if mask_np.max() > 1.0 else mask_np
                mask_accumulator += mask_np
            except Exception as e:
                print(f"    Warning: Could not process mask {mask_path}: {e}")

        im = ax.imshow(mask_accumulator, cmap='cool', interpolation='nearest')
        ax.set_title(f"'{category_to_plot}' ({num_masks} masks)")
        ax.axis("off")
        fig.colorbar(im, ax=ax, label='Overlap Count', shrink=0.8)

        plot_index += 1

    for i in range(plot_index, len(axes)):
        axes[i].axis("off")

    fig.suptitle(f"Accumulated Mask Heatmaps (Size: {image_size}x{image_size})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()