import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from natsort import natsorted
import os
import config as c
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from asdi_mri_sr.condition import ASDICondition
from asdi_mri_sr.ddim_sampler import DDIMSampler
from asdi_mri_sr.unet import ConditionalUNet

# ==================== ASDI Super-Resolution Module Configuration ====================

USE_ASDI_SR = True
ASDI_CHECKPOINT = r'./asdi_mri_sr/checkpoints/best.pt'
ASDI_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ASDI_STEPS = 20
ASDI_ETA = 0.0

# Global singleton: avoid repeated model loading
_asdi_modules = None


def _init_asdi_modules():
    """Initialize ASDI module"""
    global _asdi_modules
    if _asdi_modules is not None:
        return _asdi_modules

    device = torch.device(ASDI_DEVICE)
    unet = ConditionalUNet(latent_channels=4, condition_channels=4, base_channels=64).to(device)
    checkpoint = torch.load(ASDI_CHECKPOINT, map_location=device, weights_only=False)

    if "unet_state_dict" in checkpoint:
        state_dict = checkpoint["unet_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    cleaned = {k.replace("module.", "").replace("unet.", ""): v for k, v in state_dict.items()}
    unet.load_state_dict(cleaned, strict=False)
    unet.eval()

    conditioner = ASDICondition(upscale_factor=2).to(device)
    sampler = DDIMSampler(eta=ASDI_ETA)
    pixel_shuffle = nn.PixelShuffle(upscale_factor=2).to(device)

    _asdi_modules = (unet, conditioner, sampler, pixel_shuffle, device)
    print(f"[ASDI] Detail enhancement module loaded, device: {device}")
    return _asdi_modules


@torch.no_grad()
def asdi_enhance(image_pil):

    unet, conditioner, sampler, pixel_shuffle, device = _init_asdi_modules()
    orig_w, orig_h = image_pil.size

    img_array = np.asarray(image_pil).astype(np.float32)
    img_tensor = torch.from_numpy(img_array / 255.0)[None, None, :, :]
    img_tensor = img_tensor * 2.0 - 1.0
    img_tensor = img_tensor.to(device)

    condition, _ = conditioner(img_tensor)
    sr_latent = sampler.sample(
        condition=condition,
        unet=unet,
        num_steps=ASDI_STEPS,
        deterministic_start=True,
    )
    sr = pixel_shuffle(sr_latent)
    sr = torch.clamp(sr, -1.0, 1.0)

    sr = F.interpolate(sr, size=(orig_h, orig_w), mode='bicubic', align_corners=False)
    sr = torch.clamp(sr, -1.0, 1.0)

    sr_np = ((sr[0, 0].cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    sr_rgb = np.stack([sr_np, sr_np, sr_np], axis=-1)

    return Image.fromarray(sr_rgb, mode='RGB')


# ==================== Dataset Paths and Utility Functions ====================

TRAIN_IR_PATHS = ['./data/train/MRI-PET/MRI', './data/train/MRI-SPECT/MRI']
TRAIN_VI_PATHS = ['./data/train/MRI-PET/PET', './data/train/MRI-SPECT/SPECT']
VAL_IR_PATHS = ['./data/val/MRI-PET/MRI', './data/val/MRI-SPECT/MRI']
VAL_VI_PATHS = ['./data/val/MRI-PET/PET', './data/val/MRI-SPECT/SPECT']


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


def match_pairs(ir_paths, vi_paths):
    ir_files = []
    vi_files = []
    for ir_path, vi_path in zip(ir_paths, vi_paths):
        ir_images = natsorted(glob.glob(os.path.join(ir_path, '*')))
        vi_images = natsorted(glob.glob(os.path.join(vi_path, '*')))

        for ir_image in ir_images:
            ir_filename = os.path.basename(ir_image)
            matching_vi_image = os.path.join(vi_path, ir_filename)
            if os.path.exists(matching_vi_image):
                ir_files.append(ir_image)
                vi_files.append(matching_vi_image)
            else:
                print(f"No matching VI image found for {ir_filename}")

    return ir_files, vi_files


class Hinet_Dataset(Dataset):
    def __init__(self, transforms_=None, mode="train"):
        self.transform = transforms_
        self.mode = mode
        if mode == 'train':
            self.files1, self.files2 = match_pairs(TRAIN_IR_PATHS, TRAIN_VI_PATHS)
        else:
            self.files1, self.files2 = match_pairs(VAL_IR_PATHS, VAL_VI_PATHS)

    def __getitem__(self, index):
        max_retry = 10
        original_index = index

        for attempt in range(max_retry):
            current_index = (original_index + attempt) % len(self.files1)

            try:
                image = Image.open(self.files1[current_index])

                if USE_ASDI_SR:
                    if image.mode != 'L':
                        image = image.convert('L')
                    image = asdi_enhance(image)
                else:
                    image = to_rgb(image)

                image1 = Image.open(self.files2[current_index])
                image1 = to_rgb(image1)

                if image.size != image1.size:
                    print(f"[Warning] Size mismatch: MRI {image.size} vs PET {image1.size}, forcing PET resize")
                    image1 = image1.resize(image.size, Image.BICUBIC)

                item = self.transform(image)
                item1 = self.transform(image1)

                return item, item1

            except Exception as e:
                print(f"Error loading image at index {current_index}: {e}")

        print(f"[Critical Error] Failed to load valid sample for {max_retry} consecutive attempts, returning zero tensor")
        dummy = torch.zeros(3, 256, 256)
        return dummy, dummy

    def __len__(self):
        return len(self.files1)


# ==================== Transforms ====================

transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomCrop(c.cropsize),
    T.ToTensor()
])

transform_val = T.Compose([
    T.CenterCrop(c.cropsize_val),
    T.ToTensor(),
])


# ==================== Lazy DataLoader Creation Functions ====================

def get_trainloader():
    """Training DataLoader (lazy creation)"""
    return DataLoader(
        Hinet_Dataset(transforms_=transform, mode="train"),
        batch_size=c.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True
    )


def get_testloader():
    """Test DataLoader (lazy creation)"""
    return DataLoader(
        Hinet_Dataset(transforms_=transform_val, mode="val"),
        batch_size=c.batchsize_val,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        drop_last=True
    )