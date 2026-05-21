import torch
import torch.nn as nn
import torch.nn.functional as F
import Unet as common


class LWD(nn.Module):

    def __init__(self, in_channels):
        super(LWD, self).__init__()
        self.in_channels = in_channels
        total_channels = 4 * in_channels

        # 1x1 conv for data-driven wavelet basis update (W)
        self.weight_conv = nn.Conv2d(total_channels, total_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(total_channels)
        self.relu = nn.ReLU(inplace=True)

        # Depthwise separable conv (DWConv) for spatial adaptive threshold estimation \tau
        self.dw_conv = nn.Conv2d(total_channels, total_channels, kernel_size=3, stride=1, padding=1,
                                 groups=total_channels)

    def forward(self, x):
        # Unit-norm regularization to ensure energy conservation and filter stability
        with torch.no_grad():
            weight = self.weight_conv.weight
            norm = torch.norm(weight, p=2, dim=1, keepdim=True) + 1e-8
            self.weight_conv.weight.copy_(weight / norm)

        x_LL, x_HL, x_LH, x_HH = common.dwt_init(x)

        F_tensor = torch.cat([x_LL, x_LH, x_HL, x_HH], dim=1)

        F_hat = self.relu(self.bn(self.weight_conv(F_tensor)))

        tau = torch.abs(self.dw_conv(F_hat))
        F_tilde = torch.sign(F_hat) * torch.clamp(torch.abs(F_hat) - tau, min=0.0)

        LL, LH, HL, HH = torch.split(F_tilde, self.in_channels, dim=1)

        eps = 1e-8
        E_LL = torch.mean(torch.abs(LL))
        E_HF = torch.mean(torch.abs(LH)) + torch.mean(torch.abs(HL)) + torch.mean(torch.abs(HH))
        loss_conc = -0.05 * (E_LL / (E_HF + eps))

        return LL, LH, HL, HH, loss_conc


class LIBSubnet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(LIBSubnet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.net(x)


class LIB(nn.Module):


    def __init__(self, channels):
        super(LIB, self).__init__()
        self.phi = LIBSubnet(channels, channels)
        self.T = LIBSubnet(channels, channels)
        self.psi = LIBSubnet(channels, channels)

    def forward(self, LL_A, LL_B, rev=False):
        if not rev:
            # Forward interaction logic
            L_B = self.phi(LL_A) + LL_B
            sigma_T = torch.sigmoid(self.T(L_B))  # Use sigmoid to ensure scale stability
            L_A = LL_A * torch.exp(sigma_T) + self.psi(L_B)

            # Compute scale regularization loss to prevent excessive feature amplification
            loss_scale = 1e-3 * torch.mean(torch.abs(sigma_T))
            return L_A, L_B, loss_scale
        else:
            # Strict exact inverse reconstruction logic
            L_A, L_B = LL_A, LL_B
            sigma_T = torch.sigmoid(self.T(L_B))
            LL_A = (L_A - self.psi(L_B)) * torch.exp(-sigma_T)
            LL_B = L_B - self.phi(LL_A)
            return LL_A, LL_B