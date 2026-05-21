#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.init as init
from Feature_Extraction import LWD, LIB


# ========================== Configuration ==========================

CLAMP = 2.0


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


# ========================== ILWT ==========================

def ilwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch = in_batch
    out_channel = int(in_channel / (r ** 2))
    out_height = r * in_height
    out_width = r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class ILWT(nn.Module):
    def __init__(self):
        super(ILWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return ilwt_init(x)


# ========================== INV_block (Legacy Compatible) ==========================

class ResidualDenseBlock_out(nn.Module):
    def __init__(self, input_channels, output_channels, bias=True):
        super(ResidualDenseBlock_out, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(input_channels + 32, 32, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(input_channels + 2 * 32, 32, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(input_channels + 3 * 32, 32, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(input_channels + 4 * 32, output_channels, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        initialize_weights([self.conv5], 0.)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5


class INV_block(nn.Module):
    def __init__(self, subnet_constructor=ResidualDenseBlock_out, clamp=CLAMP, in_1=1, in_2=1):
        super().__init__()
        self.split_len1 = in_1
        self.split_len2 = in_2
        self.clamp = clamp
        self.r = subnet_constructor(self.split_len1, self.split_len2)
        self.y = subnet_constructor(self.split_len1, self.split_len2)
        self.f = subnet_constructor(self.split_len2, self.split_len1)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))
        if not rev:
            t2 = self.f(x2)
            y1 = x1 + t2
            s1, t1 = self.r(y1), self.y(y1)
            y2 = self.e(s1) * x2 + t1
        else:
            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / self.e(s1)
            t2 = self.f(y2)
            y1 = (x1 - t2)

        return y1, y2


# ========================== INet (Unified with LWD-LIB and LWD-INV) ==========================

class INet(nn.Module):
    """
    Three-level LWD-based invertible fusion network.

    Modes:
    - LWD-LIB: Paper architecture with Learned Invertible Blocks
    - LWD-INV: Legacy architecture with INV_blocks (for backward compatibility)
    """
    def __init__(self, in_c1=3, in_c2=3, block_num=[2, 1, 1], mode='LWD-LIB'):
        super(INet, self).__init__()
        self.in_c1 = in_c1
        self.in_c2 = in_c2
        self.mode = mode

        self.ilwt = ILWT()

        if mode == 'LWD-LIB':
            # Paper architecture: LWD-LIB feature extraction
            self.lwd1_A = LWD(in_c1)
            self.lwd1_B = LWD(in_c2)
            self.lib1 = LIB(in_c1)

            self.lwd2_A = LWD(in_c1)
            self.lwd2_B = LWD(in_c2)
            self.lib2 = LIB(in_c1)

            self.lwd3_A = LWD(in_c1)
            self.lwd3_B = LWD(in_c2)
            self.lib3 = LIB(in_c1)
        else:
            # Legacy architecture: LWD + INV_block
            inv_ops = []
            for i in range(block_num[0]):
                inv_ops.append(INV_block(in_1=in_c1, in_2=in_c2))
            self.inv1_ops = nn.ModuleList(inv_ops)

            inv_ops = []
            for i in range(block_num[1]):
                inv_ops.append(INV_block(in_1=in_c1, in_2=in_c2))
            self.inv2_ops = nn.ModuleList(inv_ops)

            inv_ops = []
            for i in range(block_num[2]):
                inv_ops.append(INV_block(in_1=in_c1, in_2=in_c2))
            self.inv3_ops = nn.ModuleList(inv_ops)

        # Multi-scale fusion networks (shared)
        self.cat_ll1 = nn.Conv2d(in_c1 * 2, in_c1, 1, 1, 0)
        self.cat_lh1 = nn.Conv2d(in_c1 * 2, in_c1, 1, 1, 0)
        self.cat_hl1 = nn.Conv2d(in_c1 * 2, in_c1, 1, 1, 0)
        self.cat_hh1 = nn.Conv2d(in_c1 * 2, in_c1, 1, 1, 0)

        self.cat_ll2 = nn.Conv2d(in_c1 * 2, in_c1, 1, 1, 0)
        self.cat_lh2 = nn.Conv2d(in_c1 * 2, in_c1, 1, 1, 0)
        self.cat_hl2 = nn.Conv2d(in_c1 * 2, in_c1, 1, 1, 0)
        self.cat_hh2 = nn.Conv2d(in_c1 * 2, in_c1, 1, 1, 0)

        self.cat_ll3 = nn.Conv2d(in_c1 * 2, in_c1, 1, 1, 0)
        self.cat_lh3 = nn.Conv2d(in_c1 * 2, in_c1, 1, 1, 0)
        self.cat_hl3 = nn.Conv2d(in_c1 * 2, in_c1, 1, 1, 0)
        self.cat_hh3 = nn.Conv2d(in_c1 * 2, in_c1, 1, 1, 0)

        self.up_conv1 = nn.Conv2d(in_c1, in_c1, 1, 1, 0)
        self.up_conv2 = nn.Conv2d(in_c1, in_c1, 1, 1, 0)
        self.up_conv3 = nn.Conv2d(in_c1, in_c1, 1, 1, 0)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.bn = nn.BatchNorm2d(in_c1)

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.in_c1), x.narrow(1, self.in_c1, self.in_c2))

        if not rev:
            if self.mode == 'LWD-LIB':
                return self._forward_lib(x1, x2)
            else:
                return self._forward_inv(x1, x2)
        else:
            return x

    def _forward_lib(self, x1, x2):
        """LWD-LIB forward (paper architecture)."""
        reg_loss = 0.0

        # Layer 1
        x1_LL1, x1_HL1, x1_LH1, x1_HH1, l_conc1_A = self.lwd1_A(x1)
        x2_LL1, x2_HL1, x2_LH1, x2_HH1, l_conc1_B = self.lwd1_B(x2)
        x1_LL1, x2_LL1, l_scale1 = self.lib1(x1_LL1, x2_LL1)
        reg_loss += (l_conc1_A + l_conc1_B + l_scale1)

        # Layer 2
        x1_LL2, x1_HL2, x1_LH2, x1_HH2, l_conc2_A = self.lwd2_A(x1_LL1)
        x2_LL2, x2_HL2, x2_LH2, x2_HH2, l_conc2_B = self.lwd2_B(x2_LL1)
        x1_LL2, x2_LL2, l_scale2 = self.lib2(x1_LL2, x2_LL2)
        reg_loss += (l_conc2_A + l_conc2_B + l_scale2)

        # Layer 3
        x1_LL3, x1_HL3, x1_LH3, x1_HH3, l_conc3_A = self.lwd3_A(x1_LL2)
        x2_LL3, x2_HL3, x2_LH3, x2_HH3, l_conc3_B = self.lwd3_B(x2_LL2)
        x1_LL3, x2_LL3, l_scale3 = self.lib3(x1_LL3, x2_LL3)
        reg_loss += (l_conc3_A + l_conc3_B + l_scale3)

        # Fusion
        out = self._fuse_and_reconstruct(
            x1_LL3, x1_HL3, x1_LH3, x1_HH3,
            x2_LL3, x2_HL3, x2_LH3, x2_HH3,
            x1_LL2, x1_HL2, x1_LH2, x1_HH2,
            x2_LL2, x2_HL2, x2_LH2, x2_HH2,
            x1_LL1, x1_HL1, x1_LH1, x1_HH1,
            x2_LL1, x2_HL1, x2_LH1, x2_HH1
        )

        return out, reg_loss

    def _forward_inv(self, x1, x2):
        """LWD-INV forward (legacy compatible)."""
        # Level 1
        x1_LL1, x1_HL1, x1_LH1, x1_HH1 = self._lwd_legacy(x1)
        x2_LL1, x2_HL1, x2_LH1, x2_HH1 = self._lwd_legacy(x2)
        for op in self.inv1_ops:
            x1_LL1, x2_LL1 = op.forward(torch.cat((x1_LL1, x2_LL1), dim=1))

        # Level 2
        x1_LL2, x1_HL2, x1_LH2, x1_HH2 = self._lwd_legacy(x1_LL1)
        x2_LL2, x2_HL2, x2_LH2, x2_HH2 = self._lwd_legacy(x2_LL1)
        for op in self.inv2_ops:
            x1_LL2, x2_LL2 = op.forward(torch.cat((x1_LL2, x2_LL2), dim=1))

        # Level 3
        x1_LL3, x1_HL3, x1_LH3, x1_HH3 = self._lwd_legacy(x1_LL2)
        x2_LL3, x2_HL3, x2_LH3, x2_HH3 = self._lwd_legacy(x2_LL2)
        for op in self.inv3_ops:
            x1_LL3, x2_LL3 = op.forward(torch.cat((x1_LL3, x2_LL3), dim=1))

        # Fusion
        out = self._fuse_and_reconstruct(
            x1_LL3, x1_HL3, x1_LH3, x1_HH3,
            x2_LL3, x2_HL3, x2_LH3, x2_HH3,
            x1_LL2, x1_HL2, x1_LH2, x1_HH2,
            x2_LL2, x2_HL2, x2_LH2, x2_HH2,
            x1_LL1, x1_HL1, x1_LH1, x1_HH1,
            x2_LL1, x2_HL1, x2_LH1, x2_HH1
        )

        return out

    def _lwd_legacy(self, x):
        """Legacy LWD without concatenation output."""
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4
        return x_LL, x_HL, x_LH, x_HH

    def _fuse_and_reconstruct(self,
                              x1_LL3, x1_HL3, x1_LH3, x1_HH3,
                              x2_LL3, x2_HL3, x2_LH3, x2_HH3,
                              x1_LL2, x1_HL2, x1_LH2, x1_HH2,
                              x2_LL2, x2_HL2, x2_LH2, x2_HH2,
                              x1_LL1, x1_HL1, x1_LH1, x1_HH1,
                              x2_LL1, x2_HL1, x2_LH1, x2_HH1):
        """Multi-scale fusion and reconstruction."""
        # Level 3 fusion
        matrix_w_ll = torch.sigmoid(self.cat_ll1(torch.cat((x1_LL3, x2_LL3), dim=1)))
        x_LL3 = matrix_w_ll * x1_LL3 + (1 - matrix_w_ll) * x2_LL3
        matrix_w_hl = torch.sigmoid(self.cat_hl1(torch.cat((x1_HL3, x2_HL3), dim=1)))
        x_HL3 = matrix_w_hl * x1_HL3 + (1 - matrix_w_hl) * x2_HL3
        matrix_w_lh = torch.sigmoid(self.cat_lh1(torch.cat((x1_LH3, x2_LH3), dim=1)))
        x_LH3 = matrix_w_lh * x1_LH3 + (1 - matrix_w_lh) * x2_LH3
        matrix_w_hh = torch.sigmoid(self.cat_hh1(torch.cat((x1_HH3, x2_HH3), dim=1)))
        x_HH3 = matrix_w_hh * x1_HH3 + (1 - matrix_w_hh) * x2_HH3
        x_3 = self.ilwt(torch.cat((x_LL3, x_HL3, x_LH3, x_HH3), dim=1))

        # Level 2 fusion
        matrix_w_ll = torch.sigmoid(self.cat_ll2(torch.cat((x1_LL2, x2_LL2), dim=1)))
        x_LL2 = matrix_w_ll * x1_LL2 + (1 - matrix_w_ll) * x2_LL2
        matrix_w_hl = torch.sigmoid(self.cat_hl2(torch.cat((x1_HL2, x2_HL2), dim=1)))
        x_HL2 = matrix_w_hl * x1_HL2 + (1 - matrix_w_hl) * x2_HL2
        matrix_w_lh = torch.sigmoid(self.cat_lh2(torch.cat((x1_LH2, x2_LH2), dim=1)))
        x_LH2 = matrix_w_lh * x1_LH2 + (1 - matrix_w_lh) * x2_LH2
        matrix_w_hh = torch.sigmoid(self.cat_hh2(torch.cat((x1_HH2, x2_HH2), dim=1)))
        x_HH2 = matrix_w_hh * x1_HH2 + (1 - matrix_w_hh) * x2_HH2
        x_2 = self.ilwt(torch.cat((x_LL2, x_HL2, x_LH2, x_HH2), dim=1))

        # Level 1 fusion
        matrix_w_ll = torch.sigmoid(self.cat_ll3(torch.cat((x1_LL1, x2_LL1), dim=1)))
        x_LL1 = matrix_w_ll * x1_LL1 + (1 - matrix_w_ll) * x2_LL1
        matrix_w_hl = torch.sigmoid(self.cat_hl3(torch.cat((x1_HL1, x2_HL1), dim=1)))
        x_HL1 = matrix_w_hl * x1_HL1 + (1 - matrix_w_hl) * x2_HL1
        matrix_w_lh = torch.sigmoid(self.cat_lh3(torch.cat((x1_LH1, x2_LH1), dim=1)))
        x_LH1 = matrix_w_lh * x1_LH1 + (1 - matrix_w_lh) * x2_LH1
        matrix_w_hh = torch.sigmoid(self.cat_hh3(torch.cat((x1_HH1, x2_HH1), dim=1)))
        x_HH1 = matrix_w_hh * x1_HH1 + (1 - matrix_w_hh) * x2_HH1
        x_1 = self.ilwt(torch.cat((x_LL1, x_HL1, x_LH1, x_HH1), dim=1))

        # Reconstruction
        out = self.up_conv1(self.upsample(x_3)) + x_2
        out = self.up_conv2(self.upsample(out)) + x_1
        out = self.bn(out)

        return out


class Model(nn.Module):
    """Model wrapper."""
    def __init__(self, mode='LWD-LIB'):
        super(Model, self).__init__()
        self.model = INet(in_c1=3, in_c2=3, mode=mode)

    def forward(self, x, rev=False):
        if not rev:
            out = self.model(x)
        else:
            out = self.model(x, rev=True)
        return out