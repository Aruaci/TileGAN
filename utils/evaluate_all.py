import os
import numpy as np
import torch
from skimage.io import imread
from skimage.transform import resize
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import lpips
import cv2
from pytorch_fid import fid_score
import clip
from PIL import Image

# Initialize LPIPS
lpips_model = lpips.LPIPS(net='alex')
lpips_model.eval()

# Initialize CLIP
clip_model, clip_preprocess = clip.load("ViT-B/32")
clip_model.eval()

def load_image(path, size=(256, 256), grayscale=False):
    img = imread(path)
    img = resize(img, size, anti_aliasing=True)
    if grayscale and img.ndim == 3:
        img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return img


def mask_only(img, mask):
    return img * mask


def save_masked_patches(gt_dir, gen_dir, mask_dir, save_real_dir, save_fake_dir, size=(256, 256)):
    os.makedirs(save_real_dir, exist_ok=True)
    os.makedirs(save_fake_dir, exist_ok=True)

    gt_files = sorted(os.listdir(gt_dir))
    gen_files = sorted(os.listdir(gen_dir))
    mask_files = sorted(os.listdir(mask_dir))

    for idx, (gt_file, gen_file, mask_file) in enumerate(zip(gt_files, gen_files, mask_files)):
        gt = load_image(os.path.join(gt_dir, gt_file), size=size, grayscale=False)
        gen = load_image(os.path.join(gen_dir, gen_file), size=size, grayscale=False)
        mask = load_image(os.path.join(mask_dir, mask_file), size=size, grayscale=True)

        mask = (mask > 0.5).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)

        real_patch = (gt * mask).astype(np.float32)
        fake_patch = (gen * mask).astype(np.float32)

        real_path = os.path.join(save_real_dir, f"{idx:03d}.png")
        fake_path = os.path.join(save_fake_dir, f"{idx:03d}.png")

        cv2.imwrite(real_path, (real_patch * 255).astype(np.uint8))
        cv2.imwrite(fake_path, (fake_patch * 255).astype(np.uint8))


def evaluate_inpainting(gt_dir, gen_dir, mask_dir, size=(256, 256)):
    gt_files = sorted(os.listdir(gt_dir))
    gen_files = sorted(os.listdir(gen_dir))
    mask_files = sorted(os.listdir(mask_dir))

    assert len(gt_files) == len(gen_files) == len(mask_files), "Mismatch in dataset lengths"

    total_lpips = []
    total_l1 = []
    total_ssim = []
    total_clip_scores = []

    text_prompt = "a floor tile damaged with a realistic fracture"
    text_tokens = clip.tokenize([text_prompt]).cuda() if torch.cuda.is_available() else clip.tokenize([text_prompt])
    
    for gt_file, gen_file, mask_file in tqdm(zip(gt_files, gen_files, mask_files), total=len(gt_files)):
        gt = load_image(os.path.join(gt_dir, gt_file), size=size, grayscale=False)
        gen = load_image(os.path.join(gen_dir, gen_file), size=size, grayscale=False)
        mask = load_image(os.path.join(mask_dir, mask_file), size=size, grayscale=True)

        mask = (mask > 0.5).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)

        ssim_score = ssim(gt, gen, channel_axis=-1, data_range=1.0)
        total_ssim.append(ssim_score)

        masked_l1 = np.mean(np.abs((gt - gen) * mask))
        total_l1.append(masked_l1)

        gt_tensor = torch.tensor(gt.transpose(2, 0, 1)).unsqueeze(0).float()
        gen_tensor = torch.tensor(gen.transpose(2, 0, 1)).unsqueeze(0).float()
        mask_tensor = torch.tensor(mask.transpose(2, 0, 1)).unsqueeze(0).float()

        lpips_score = lpips_model(gt_tensor * mask_tensor, gen_tensor * mask_tensor).item()
        total_lpips.append(lpips_score)

        # CLIP Score (image-text similarity)
        pil_img = Image.fromarray((gen * 255).astype(np.uint8))
        clip_input = clip_preprocess(pil_img).unsqueeze(0)
        if torch.cuda.is_available():
            clip_input = clip_input.cuda()
            clip_model.cuda()

        with torch.no_grad():
            image_features = clip_model.encode_image(clip_input)
            text_features = clip_model.encode_text(text_tokens)
            clip_score = torch.nn.functional.cosine_similarity(image_features, text_features).item()
            total_clip_scores.append(clip_score)

    print("\nEvaluation Results:")
    print("Average SSIM:", np.mean(total_ssim))
    print("Average Masked L1:", np.mean(total_l1))
    print("Average LPIPS:", np.mean(total_lpips))
    print("Average CLIP score:", np.mean(total_clip_scores))

    real_patch_dir = "./tmp_fid1/real"
    fake_patch_dir = "./tmp_fid1/fake"
    save_masked_patches(gt_dir, gen_dir, mask_dir, real_patch_dir, fake_patch_dir, size=size)

    fid_value = fid_score.calculate_fid_given_paths([real_patch_dir, fake_patch_dir], batch_size=16, device='cpu', dims=2048)
    print("FID on masked regions:", fid_value)


if __name__ == "__main__":
    ground_truth_dir = "./tile/test/crack"
    generated_dir = "./generated_images/fake"
    mask_dir = "./tile/ground_truth/crack"

    evaluate_inpainting(ground_truth_dir, generated_dir, mask_dir, size=(256, 256))
