#!/usr/bin/env python3
"""
Inference for medical image fusion.
Supported tasks: MRI-PET, MRI-SPECT, CT-PET
"""

import os
import time
import glob
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
from torchvision.transforms import ToTensor
from SuperRev_Fusion import Model


# ========================== Configuration (Edit Here) ==========================

TASK = 'MRI-PET'              # Options: 'MRI-PET', 'MRI-SPECT', 'CT-PET'
DATA_ROOT = './test_data'     # Dataset root directory
MODEL_PATH = './checkpoints/model-MRI.pt'     # options: model-CT.pt/model-MRI.pt
OUTPUT_DIR = './results'      # Output directory
DEVICE = 'cuda:0'             # Device: 'cuda:0' or 'cpu'



# ========================== Task Definitions ==========================

TASKS = {
    'MRI-PET': {'A': 'MRI', 'B': 'PET'},
    'MRI-SPECT': {'A': 'MRI', 'B': 'SPECT'},
    'CT-PET': {'A': 'CT', 'B': 'PET'},
}


# ========================== Legacy Compatibility ==========================

INIT_SCALE = 0.01
ARCH_MODE = 'LWD-INV'


def init_model(mod):
    """Initialize model parameters (legacy compatible)."""
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            param.data = INIT_SCALE * torch.randn(param.data.shape).cuda()
            if split[-2] == 'conv5':
                param.data.fill_(0.)


def strip_module_prefix(state_dict):
    """Remove 'module.' prefix for DataParallel compatibility."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


# ========================== Test Functions ==========================

def load_model(model_path, device, mode='LWD-LIB'):
    """Load pretrained model with specified architecture mode."""
    net = Model(mode=mode)
    net.to(device)
    init_model(net)

    state_dicts = torch.load(model_path, map_location=device, weights_only=True)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    network_state_dict = strip_module_prefix(network_state_dict)
    net.load_state_dict(network_state_dict)
    net.eval()
    return net


def preprocess(img_path, device):
    """Load and preprocess image to tensor."""
    img = Image.open(img_path).convert("RGB")
    tensor = ToTensor()(img).unsqueeze(0).to(device)
    return tensor


def fuse(net, img_A, img_B):
    """Run fusion inference."""
    with torch.no_grad():
        input_img = torch.cat((img_A, img_B), 1)
        output = net(input_img)
    return output


def run_test(task, data_root, model_path, output_dir, device, mode='LWD-LIB'):
    """Run fusion test for specified task and architecture mode."""
    task_info = TASKS[task]
    dir_A = os.path.join(data_root, task, task_info['A'])
    dir_B = os.path.join(data_root, task, task_info['B'])
    out_dir = os.path.join(output_dir, task)
    os.makedirs(out_dir, exist_ok=True)

    net = load_model(model_path, device, mode)

    exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif']
    files_A = []
    for ext in exts:
        files_A.extend(glob.glob(os.path.join(dir_A, ext)))
        files_A.extend(glob.glob(os.path.join(dir_A, ext.upper())))
    files_A = sorted(list(set(files_A)))

    if not files_A:
        print(f"No images found in {dir_A}")
        return

    print(f"Task: {task}, Mode: {mode}, Images: {len(files_A)}")
    times = []

    for i, path_A in enumerate(files_A):
        name = os.path.basename(path_A)
        path_B = os.path.join(dir_B, name)

        if not os.path.exists(path_B):
            print(f"Skip {name}: not found in {dir_B}")
            continue

        img_A = preprocess(path_A, device)
        img_B = preprocess(path_B, device)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        tic = time.time()

        output = fuse(net, img_A, img_B)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.time() - tic
        times.append(elapsed)

        out_path = os.path.join(out_dir, name)
        torchvision.utils.save_image(output, out_path)

        print(f"[{i+1}/{len(files_A)}] {name}  Time: {elapsed*1000:.2f}ms")

    if len(times) > 4:
        avg = sum(times[2:-2]) / len(times[2:-2])
    else:
        avg = sum(times) / len(times)
    print(f"Average time: {avg*1000:.2f}ms")
    print(f"Results saved to: {out_dir}")


# ========================== Main ==========================

def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Architecture mode: {ARCH_MODE}")
    print(f"Model path: {MODEL_PATH}")

    run_test(TASK, DATA_ROOT, MODEL_PATH, OUTPUT_DIR, device, ARCH_MODE)
    print("Done.")


if __name__ == '__main__':
    main()