#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim
import math
import numpy as np
import warnings
import logging
import time
import os

from SuperRev_Fusion import Model, init_model
import config as c
from tensorboardX import SummaryWriter
import datasets
import viz
import util
from pytorch_ssim import ssim, gradient, Fusionloss

warnings.filterwarnings("ignore")
device = torch.device(f"cuda:{str(c.device_ids[0])}" if torch.cuda.is_available() else "cpu")


def gauss_noise(shape):
    """Generate Gaussian noise for regularization."""
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()
    return noise


def guide_loss(output, bicubic_image):
    """Guidance loss using MSE for chrominance channels."""
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
    loss = loss_fn(output, bicubic_image)
    return loss.cuda()


def reconstruction_loss(rev_input, input):
    """Reconstruction loss for invertible path validation."""
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
    loss = loss_fn(rev_input, input)
    return loss.cuda()


def low_frequency_loss(ll_input, gt_input):
    """Low-frequency preservation loss for wavelet subbands."""
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
    loss = loss_fn(ll_input, gt_input)
    return loss.cuda()


def get_parameter_number(net):
    """Count total and trainable parameters in the network."""
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def computePSNR(origin, pred):
    """Compute Peak Signal-to-Noise Ratio."""
    origin = np.array(origin).astype(np.float32)
    pred = np.array(pred).astype(np.float32)
    mse = np.mean((origin / 1.0 - pred / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)


def AG(img):
    """Average Gradient metric for spatial quality assessment."""
    Gx, Gy = np.zeros_like(img), np.zeros_like(img)
    Gx[:, 0] = img[:, 1] - img[:, 0]
    Gx[:, -1] = img[:, -1] - img[:, -2]
    Gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2
    Gy[0, :] = img[1, :] - img[0, :]
    Gy[-1, :] = img[-1, :] - img[-2, :]
    Gy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2
    return np.mean(np.sqrt((Gx ** 2 + Gy ** 2) / 2))


def CC(image_F, image_A, image_B):
    """Correlation Coefficient between fused and source images."""
    rAF = np.sum((image_A - np.mean(image_A)) * (image_F - np.mean(image_F))) / np.sqrt(
        (np.sum((image_A - np.mean(image_A)) ** 2)) * (np.sum((image_F - np.mean(image_F)) ** 2)))
    rBF = np.sum((image_B - np.mean(image_B)) * (image_F - np.mean(image_F))) / np.sqrt(
        (np.sum((image_B - np.mean(image_B)) ** 2)) * (np.sum((image_F - np.mean(image_F)) ** 2)))
    return (rAF + rBF) / 2


def SCD(image_F, image_A, image_B):
    """Sum of Correlation Differences for fusion quality."""
    imgF_A = image_F - image_A
    imgF_B = image_F - image_B
    corr1 = np.sum((image_A - np.mean(image_A)) * (imgF_B - np.mean(imgF_B))) / np.sqrt(
        (np.sum((image_A - np.mean(image_A)) ** 2)) * (np.sum((imgF_B - np.mean(imgF_B)) ** 2)))
    corr2 = np.sum((image_B - np.mean(image_B)) * (imgF_A - np.mean(imgF_A))) / np.sqrt(
        (np.sum((image_B - np.mean(image_B)) ** 2)) * (np.sum((imgF_A - np.mean(imgF_A)) ** 2)))
    return corr1 + corr2


def evaluator(A, B, F):
    """Comprehensive fusion quality evaluator."""
    A = np.array(A).astype(np.float32)
    B = np.array(B).astype(np.float32)
    F = np.array(F).astype(np.float32)
    SF = np.sqrt(np.mean((F[:, 1:] - F[:, :-1]) ** 2) + np.mean((F[1:, :] - F[:-1, :]) ** 2))
    SD = np.std(F)
    AG_f = AG(F)
    CC_ALL = CC(F, A, B)
    SCD_ALL = SCD(F, A, B)
    return SD + CC_ALL + SCD_ALL, SF + AG_f


def load(name):
    """Load model checkpoint."""
    state_dicts = torch.load(name, map_location=device)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')


def rgb2ycbcr_t(img_rgb):
    """Convert RGB tensor to YCbCr color space (batch version)."""
    R = img_rgb[:, 0, :, :].unsqueeze(1)
    G = img_rgb[:, 1, :, :].unsqueeze(1)
    B = img_rgb[:, 2, :, :].unsqueeze(1)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128 / 255
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128 / 255
    img_ycbcr = torch.cat([Y, Cb, Cr], axis=1)
    return img_ycbcr, Y, Cb, Cr


def rgb2ycbcr_t1(img_rgb):
    """Convert RGB numpy array to YCbCr (single image version)."""
    R = img_rgb[0, :, :]
    G = img_rgb[1, :, :]
    B = img_rgb[2, :, :]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128 / 255
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128 / 255
    return Y, Y, Cb, Cr


def Grad_loss(output):
    """Gradient loss for edge preservation."""
    out_grad = torch.mean(gradient(output), dim=[1, 2, 3])
    out_grad = 1 - torch.mean(out_grad / (out_grad + 1.0))
    return out_grad


# =============================================================================
# Model and Optimizer Initialization
# =============================================================================

loss_func = Fusionloss().cuda()
net = Model()
net.cuda()
init_model(net)
net = torch.nn.DataParallel(net, device_ids=c.device_ids)
para = get_parameter_number(net)
print(f"[Model] Parameter count: {para}")

params_trainable = list(filter(lambda p: p.requires_grad, net.parameters()))
optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)

if c.tain_next:
    load(c.MODEL_PATH + c.suffix)
    optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)

util.setup_logger('train', './logging/', 'train_', level=logging.INFO, screen=True, tofile=True)
logger_train = logging.getLogger('train')

localtime = time.strftime("%Y%m%d_%H%M", time.localtime())
param_loss = f"a{str(c.l_alpha)}_b{str(c.l_beta)}_c{str(c.l_gamma)}_d{str(c.l_ks)}"
SAVE_PATH = c.MODEL_PATH + f'{param_loss}/{localtime}_{str(c.mse_w)}/'
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

logger_train.info(f"Loss param: alpha={c.l_alpha}, beta={c.l_beta}, gamma={c.l_gamma}, kesa={c.l_ks}, mse_w={c.mse_w}")

# =============================================================================
# Training Main Loop
# =============================================================================

if __name__ == '__main__':
    try:
        writer = SummaryWriter(comment='hinet', filename_suffix="steg")

        # Data loaders
        trainloader = datasets.get_trainloader()
        testloader = datasets.get_testloader()

        MAX_SD = 0
        MAX_AG = 0
        sum_AS = 0

        for i_epoch in range(c.epochs):
            i_epoch = i_epoch + c.trained_epoch + 1
            loss_history = []
            g_loss_history = []
            r_loss_history = []
            l_loss_history = []

            # Training phase
            net.train()
            for i_batch, (data0, data1) in enumerate(trainloader):
                cover = data0.to(device)
                secret = data1.to(device)
                input_img = torch.cat((cover, secret), 1)

                # Forward: output and reg_loss from LWD-LIB architecture
                output, reg_loss = net(input_img)

                # Loss computation
                FUSION_ycbcr, FUSION_Y, FUSION_Cb, FUSION_Cr = rgb2ycbcr_t(output)
                OTHER_ycbcr, OTHER_Y, OTHER_Cb, OTHER_Cr = rgb2ycbcr_t(secret)
                MRI_Y = cover[:, 0, :, :].unsqueeze(1)

                LOSS_SSIM = 1 - ssim(FUSION_Y, MRI_Y, OTHER_Y)
                mse_chro = guide_loss(FUSION_Cb, OTHER_Cb) + guide_loss(FUSION_Cr, OTHER_Cr)
                mse_Y = guide_loss(FUSION_Y, MRI_Y) + c.mse_w * guide_loss(FUSION_Y, OTHER_Y)
                grad_loss = Grad_loss(FUSION_Y)

                l_loss = c.l_gamma * mse_Y + c.l_ks * mse_chro
                g_loss = grad_loss
                r_loss = LOSS_SSIM

                # Total loss with regularization from LIB blocks
                total_loss = c.l_alpha * r_loss + c.l_beta * g_loss + l_loss + torch.mean(reg_loss)

                total_loss.backward()
                optim.step()
                optim.zero_grad()

                loss_history.append([total_loss.item(), 0.])
                g_loss_history.append([g_loss.item(), 0.])
                r_loss_history.append([r_loss.item(), 0.])
                l_loss_history.append([l_loss.item(), 0.])

            epoch_losses = np.mean(np.array(loss_history), axis=0)
            r_epoch_losses = np.mean(np.array(r_loss_history), axis=0)
            g_epoch_losses = np.mean(np.array(g_loss_history), axis=0)
            l_epoch_losses = np.mean(np.array(l_loss_history), axis=0)

            # Validation phase
            psnr_s = []
            psnr_c = []
            if i_epoch % c.val_freq == 0:
                with torch.no_grad():
                    net.eval()
                    for (x0, x1) in testloader:
                        cover = x0.cuda()
                        secret = x1.cuda()
                        input_img = torch.cat((cover, secret), 1)

                        # Validation ignores reg_loss
                        output, _ = net(input_img)

                        output = output.cpu().numpy().squeeze() * 255
                        np.clip(output, 0, 255)
                        secret = secret.cpu().numpy().squeeze() * 255
                        np.clip(secret, 0, 255)
                        cover = cover.cpu().numpy().squeeze() * 255
                        np.clip(cover, 0, 255)

                        _, output_Y, _, _ = rgb2ycbcr_t1(output)
                        _, secret_Y, _, _ = rgb2ycbcr_t1(secret)
                        sd_et, ag_et = evaluator(secret_Y, cover, output_Y)
                        psnr_s.append(sd_et)
                        psnr_c.append(ag_et)

                    writer.add_scalar("SD+CC+SCD", np.mean(psnr_s), i_epoch)
                    writer.add_scalar("SF+AG", np.mean(psnr_c), i_epoch)
                    logger_train.info(
                        f"TEST: SD+CC+SCD: {np.mean(psnr_s):.4f} | SF+AG: {np.mean(psnr_c):.4f}"
                    )

            # Log training losses
            writer.add_scalars("Train", {"Train_Loss": epoch_losses[0]}, i_epoch)
            logger_train.info(
                f"Epoch {i_epoch}: Loss={epoch_losses[0].item():.4f} | "
                f"r_Loss={r_epoch_losses[0].item():.4f} | "
                f"g_Loss={g_epoch_losses[0].item():.4f} | "
                f"l_Loss={l_epoch_losses[0].item():.4f} | "
                f"LR={optim.param_groups[0]['lr']:.8f}"
            )

            # Save best model
            if i_epoch > 600 and (MAX_SD < np.mean(psnr_s) or MAX_AG < np.mean(psnr_c) or 
                                  sum_AS < (np.mean(psnr_s) + np.mean(psnr_c))) and                     np.mean(psnr_c) < 50 and np.mean(psnr_s) < 90:
                if MAX_SD < np.mean(psnr_s):
                    MAX_SD = np.mean(psnr_s)
                if MAX_AG < np.mean(psnr_c):
                    MAX_AG = np.mean(psnr_c)
                if sum_AS < (np.mean(psnr_s) + np.mean(psnr_c)):
                    sum_AS = np.mean(psnr_s) + np.mean(psnr_c)

                torch.save({
                    'opt': optim.state_dict(),
                    'net': net.state_dict()
                }, SAVE_PATH + f'model_checkpoint_{i_epoch}_{np.mean(psnr_s):.4f}_{np.mean(psnr_c):.4f}.pt')

            weight_scheduler.step()

            # Save latest model
            torch.save({
                'opt': optim.state_dict(),
                'net': net.state_dict()
            }, SAVE_PATH + 'model.pt')

        writer.close()

    except Exception as e:
        logger_train.error(f"Training error: {e}")
        if c.checkpoint_on_error:
            torch.save({
                'opt': optim.state_dict(),
                'net': net.state_dict()
            }, SAVE_PATH + 'model_ABORT.pt')
        raise

    finally:
        viz.signal_stop()
