import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import os
import torchvision.utils as vutils
import math

def train_conditional_gan(
    generator,
    discriminator,
    dataloader,
    optimizer_G,
    optimizer_D,
    adversarial_loss,
    reconstruction_loss,
    lambda_l1,
    epochs,
    device,
    fixed_batch_for_vis=None,
    checkpoint_dir=None,
    sample_dir=None,
    save_checkpoint_freq=10, 
    save_samples_freq=1
    ):
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Created checkpoint directory: {checkpoint_dir}")
    if sample_dir and not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
        print(f"Created sample directory: {sample_dir}")

    print(f"Starting Conditional GAN Training for {epochs} epochs on {device}...")
    generator.to(device)
    discriminator.to(device)

    fixed_masked_input, fixed_target, fixed_mask, fixed_condition = (None,) * 4
    if fixed_batch_for_vis:
        fixed_masked_input = fixed_batch_for_vis[0].to(device)
        fixed_mask = fixed_batch_for_vis[2].to(device)
        fixed_condition = fixed_batch_for_vis[3].to(device)
        print(f"Using fixed batch of size {fixed_masked_input.size(0)} for visualization.")

    # --- Initialize tqdm outside the epoch loop ---
    total_steps = len(dataloader) * epochs
    progress_bar = tqdm(range(total_steps), desc="Total Training Progress")
    # ---------------------------------------------

    # --- Training Loop ---
    for epoch in range(epochs):
        generator.train()
        discriminator.train()

        # No tqdm wrapper here anymore
        for i, (masked_input, target_image, mask, condition_label) in enumerate(dataloader):

            # Move data to device
            masked_input = masked_input.to(device)
            target_image = target_image.to(device)
            mask = mask.to(device)
            condition_label = condition_label.to(device)
            batch_size = masked_input.size(0)

            # Adversarial ground truths
            real_label_val = 1.0
            fake_label_val = 0.0
            real_labels = torch.full((batch_size,), real_label_val, dtype=torch.float, device=device)
            fake_labels = torch.full((batch_size,), fake_label_val, dtype=torch.float, device=device)

            # ======================== #
            # Train Discriminator      #
            # ======================== #
            discriminator.zero_grad()
            # Real images
            output_real = discriminator(target_image, condition_label)
            real_labels_shaped = torch.full_like(output_real, real_label_val, device=device) # Create shaped labels
            errD_real = adversarial_loss(output_real, real_labels_shaped)
            # Fake images
            with torch.no_grad():
                fake_images = generator(masked_input, mask, condition_label).detach()
            output_fake = discriminator(fake_images, condition_label)
            fake_labels_shaped = torch.full_like(output_fake, fake_label_val, device=device) # Create shaped labels
            errD_fake = adversarial_loss(output_fake, fake_labels_shaped)
            # Update D
            errD = (errD_real + errD_fake) / 2
            errD.backward()
            optimizer_D.step()
            D_x = output_real.mean().item()
            D_G_z1 = output_fake.mean().item()

            # ======================== #
            # Train Generator          #
            # ======================== #
            generator.zero_grad()
            # Generate fake images
            fake_images_for_G = generator(masked_input, mask, condition_label)
            # Get D's opinion
            output_G = discriminator(fake_images_for_G, condition_label)
            # Adversarial Loss (use shaped real labels)
            real_labels_for_G = torch.full_like(output_G, real_label_val, device=device)
            errG_adv = adversarial_loss(output_G, real_labels_for_G)
            # Reconstruction Loss (L1)
            errG_l1 = reconstruction_loss(fake_images_for_G * mask, target_image * mask)
            # Total G Loss
            errG = errG_adv + lambda_l1 * errG_l1
            # Update G
            errG.backward()
            optimizer_G.step()
            D_G_z2 = output_G.mean().item()

            # --- Update the single progress bar ---
            progress_bar.update(1)
            if i % 10 == 0: # Update postfix less frequently if needed
                progress_bar.set_postfix(Epoch=f"{epoch+1}/{epochs}", Loss_D=f"{errD.item():.4f}", Loss_G=f"{errG.item():.4f}", L1=f"{errG_l1.item():.4f}")
            # --------------------------------------

        # --- End of Epoch ---
        # Optional: Print summary at end of epoch if desired
        # print(f"Epoch {epoch+1}/{epochs} finished. Last Batch Losses: D={errD.item():.4f}, G={errG.item():.4f}")

        # --- Saving Samples/Checkpoints (Logic remains similar) ---
        if sample_dir and fixed_batch_for_vis and (epoch + 1) % save_samples_freq == 0:
             with torch.no_grad():
                 generator.eval()
                 fixed_fake_samples = generator(fixed_masked_input, fixed_mask, fixed_condition).detach().cpu()
                 vutils.save_image(fixed_fake_samples,
                                   os.path.join(sample_dir, f"conditional_gan_samples_epoch_{epoch+1:03d}.png"),
                                   normalize=True,
                                   nrow=int(math.sqrt(fixed_masked_input.size(0))))
                 generator.train()
             # print(f"Saved sample images for epoch {epoch+1} to {sample_dir}") # Optional print

        if checkpoint_dir and (epoch + 1) % save_checkpoint_freq == 0:
            gen_path = os.path.join(checkpoint_dir, f"cgan_generator_epoch_{epoch+1:03d}.pth")
            disc_path = os.path.join(checkpoint_dir, f"cgan_discriminator_epoch_{epoch+1:03d}.pth")
            torch.save(generator.state_dict(), gen_path)
            torch.save(discriminator.state_dict(), disc_path)
            # print(f"Saved model checkpoints for epoch {epoch+1} to {checkpoint_dir}") # Optional print

    progress_bar.close() # Close the progress bar cleanly after all epochs
    print(f"Training Finished after {epochs} epochs.")