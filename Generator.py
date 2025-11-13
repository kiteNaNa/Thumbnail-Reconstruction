import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class CCM(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.ccm = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.ccm(x)

class downsample_vit(nn.Module):
    def __init__(self,
                 dim,
                 window_size=8,
                 attn_drop=0.,
                 proj_drop=0.,
                 down_scale=2, ):
        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.scale = dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def window_partition(self, x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

    def window_reverse(self, windows, window_size, h, w):
        """
        Args:
            windows: (num_windows*b, window_size, window_size, c)
            window_size (int): Window size
            h (int): Height of image
            w (int): Width of image

        Returns:
            x: (b, h, w, c)
        """
        b = int(windows.shape[0] / (h * w / window_size / window_size))
        x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.window_size, self.window_size
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'

        lepe = func(x)  ### B', C, H', W'
        lepe = lepe.reshape(-1, C, H_sp * W_sp).permute(0, 2, 1).contiguous()

        x = x.reshape(-1, C, H_sp * W_sp).permute(0, 2, 1).contiguous()
        return x, lepe

    def forward(self, x):
        B, C, H, W = x.shape

        ################################
        # 1. window partition
        ################################
        x = x.permute(0, 2, 3, 1)
        x_window = self.window_partition(x, self.window_size).permute(0, 3, 1, 2)
        x_window = x_window.permute(0, 2, 3, 1).view(-1, self.window_size * self.window_size, C)

        ################################
        # 2. make qkv
        ################################
        qkv = self.qkv(x_window)
        # qkv = qkv.permute(0,2,3,1)
        # qkv = qkv.reshape(-1, self.window_size * self.window_size, 3*C)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        ################################
        # 3. attn and PE
        ################################
        v, lepe = self.get_lepe(v, self.get_v)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        # x = x.reshape(-1, self.window_size, self.window_size, C)
        # x = x.permute(0,3,1,2)

        ################################
        # 4. proj and drop
        ################################
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.reshape(-1, self.window_size, self.window_size, C)
        x = self.window_reverse(x, self.window_size, H, W)

        return x.permute(0, 3, 1, 2)

class LHSB(nn.Module):
    def __init__(self,
                 dim,
                 attn_drop=0.,
                 proj_drop=0.,
                 n_levels=4,
                 window_size = 8,):

        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels

        # Spatial Weighting
        self.mfr = nn.ModuleList([
            downsample_vit(dim // 4,
                           window_size=window_size,
                           attn_drop=attn_drop,
                           proj_drop=proj_drop,
                           down_scale=2 ** i)
            for i in range(self.n_levels)])

        # # Feature Aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

        # Activation
        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]

        xc = x.chunk(self.n_levels, dim=1)
        SA_before_idx = None
        out = []

        downsampled_feat = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h // 2 ** i, w // 2 ** i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                downsampled_feat.append(s)

            else:
                downsampled_feat.append(xc[i])

        for i in reversed(range(self.n_levels)):
            s = self.mfr[i](downsampled_feat[i])
            s_upsample = F.interpolate(s, size=(s.shape[2] * 2, s.shape[3] * 2), mode='nearest')

            if i > 0:
                downsampled_feat[i - 1] = downsampled_feat[i - 1] + s_upsample

            s_original_shape = F.interpolate(s, size=(h, w), mode='nearest')
            out.append(s_original_shape)

        out = self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * x
        return out

class AttBlock(nn.Module):
    def __init__(self,
                 dim,
                 ffn_scale=2.0,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 window_size = 8,):
        super().__init__()

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        # Multiscale Block
        self.lhsb = LHSB(dim,
                         attn_drop=attn_drop,
                         proj_drop=drop,
                         window_size=window_size)

        # Feedforward layer
        self.ccm = CCM(dim, ffn_scale)

    def forward(self, x):
        x = self.lhsb(self.norm1(x)) + x
        x = self.ccm(self.norm2(x)) + x
        return x


class Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False, padding_mode='reflect')
            if down
            else torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.ReLU() if act == "relu" else torch.nn.LeakyReLU(negative_slope=0.2)
        )
        self.use_dropout = use_dropout
        self.dropout = torch.nn.Dropout(p=0.5)
    def forward(self,x):
        x = self.conv(x)
        x = self.dropout(x) if self.use_dropout else x
        return x

class Residual(torch.nn.Module):
    def __init__(self, in_channels, out_channels, use_1conv=False):
        super(Residual, self).__init__()
        self.ReLU = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='reflect')
        self.conv2 = torch.nn.Conv2d(in_channels=out_channels,  out_channels=out_channels, kernel_size=3, stride=1, padding =1)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        if use_1conv:
            self.conv3 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        else:
            self.conv3 = None
    def forward(self, x):
        y = self.ReLU(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y = y+x
        return y


class Generator(torch.nn.Module):
    def __init__(self, in_channles=3, features=64):
        super(Generator, self).__init__()
        self.initial_down = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channles,out_channels=features,kernel_size=(4,4), stride=(2,2),padding=(1,1),padding_mode='reflect'),
            torch.nn.LeakyReLU(negative_slope=0.2)
        )

        self.down1 = Block(in_channels=features*1, out_channels=features*2, down=True, act='leaky', use_dropout=False)
        self.down2 = Block(in_channels=features*2, out_channels=features*4, down=True, act='leaky', use_dropout=False)
        dpr = [x.item() for x in torch.linspace(0, 0, 8)]  # stochastic depth decay rule
        self.attn1 = torch.nn.Sequential(*[AttBlock(features*4,
                                                   ffn_scale=2.0,
                                                   drop=0.,
                                                   attn_drop=0.,
                                                   drop_path=dpr[i],
                                                    window_size=2
                                                   )
                                          for i in range(4)])
        self.down3 = Block(in_channels=features*4, out_channels=features*8, down=True, act='leaky', use_dropout=False)


        self.b1 = Residual(in_channels=features * 8, out_channels=features * 8, use_1conv=False)
        self.b2 = Residual(in_channels=features * 8, out_channels=features * 8, use_1conv=False)
        self.b3 = Residual(in_channels=features * 8, out_channels=features * 8, use_1conv=False)
        self.b4 = Residual(in_channels=features * 8, out_channels=features * 8, use_1conv=False)
        self.b5 = Residual(in_channels=features * 8, out_channels=features * 8, use_1conv=False)
        self.b6 = Residual(in_channels=features * 8, out_channels=features * 8, use_1conv=False)


        self.up5 = Block(in_channels=features*8*2, out_channels=features*4, down=False, act="relu", use_dropout=False)
        self.up6 = Block(in_channels=features*4*2, out_channels=features*2, down=False, act="relu", use_dropout=False)
        self.attn2 = torch.nn.Sequential(*[AttBlock(features * 2,
                                                   ffn_scale=2.0,
                                                   drop=0.,
                                                   attn_drop=0.,
                                                   drop_path=dpr[i],
                                                    window_size=4,
                                                   )
                                          for i in range(4)])
        self.up7 = Block(in_channels=features*2*2, out_channels=features*1, down=False, act="relu", use_dropout=False)
        self.final_up = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=features*2, out_channels=in_channles, kernel_size=(4,4), stride=(2,2), padding=(1,1)),
            torch.nn.Tanh()
        )

    def cosin_metric(self, x1, x2):
        return torch.sum(x1 * x2, dim=1) / (torch.norm(x1, dim=1) * torch.norm(x2, dim=1))

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        a1 = self.attn1(d3) + d3

        d4 = self.down3(a1)

        r1 = self.b1(d4)
        r2 = self.b2(r1)
        r3 = self.b3(r2)
        r4 = self.b4(r3)
        r5 = self.b5(r4)
        r6 = self.b6(r5)


        u5 = self.up5(torch.cat([r6, d4], dim=1))
        u6 = self.up6(torch.cat([u5, a1], dim=1))
        a2 = self.attn2(u6) + u6

        u7 = self.up7(torch.cat([a2, d2], dim=1))
        final_up = self.final_up(torch.cat([u7, d1], dim=1))
        return final_up

