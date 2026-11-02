"""Experimental 2D ConvNeXt/CSWin segmenter, not the authors' released model.

CSWin follows horizontal/vertical stripe attention and window-local positional
encoding from https://github.com/microsoft/CSWin-Transformer. Attention uses
PyTorch SDPA; ConvNeXt and stochastic depth use timm.
"""
import torch
from torch import nn
from torch.nn import functional as F
from timm.layers import DropPath
import timm


class StripeAttention(nn.Module):
    def __init__(self, dim, heads, width, horizontal):
        super().__init__()
        self.heads, self.width, self.horizontal = heads, width, horizontal
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, q, k, v):
        b, c, h, w = q.shape
        sh, sw = (self.width, w) if self.horizontal else (h, self.width)
        if h % sh or w % sw:
            raise ValueError("Feature dimensions must be divisible by stripe width")

        def windows(x):
            return x.reshape(b, c, h // sh, sh, w // sw, sw).permute(
                0, 2, 4, 1, 3, 5).reshape(-1, c, sh, sw)

        def tokens(x):
            return x.flatten(2).reshape(-1, self.heads, c // self.heads, sh * sw).transpose(-1, -2)

        qw, kw, vw = windows(q), windows(k), windows(v)
        a = F.scaled_dot_product_attention(tokens(qw), tokens(kw), tokens(vw))
        a = a.transpose(-1, -2).reshape(-1, c, sh, sw) + self.lepe(vw)
        return a.reshape(b, h // sh, w // sw, c, sh, sw).permute(
            0, 3, 1, 4, 2, 5).reshape(b, c, h, w)


class CSWinBlock(nn.Module):
    def __init__(self, dim, heads, stripe, last=False, drop=0.1):
        super().__init__()
        self.last = last
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.attns = nn.ModuleList(
            [StripeAttention(dim, heads, stripe, True)] if last else
            [StripeAttention(dim // 2, heads // 2, stripe, horizontal) for horizontal in (True, False)])
        self.proj = nn.Conv2d(dim, dim, 1)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(dim * 4, dim), nn.Dropout(drop))
        self.path = DropPath(drop)

    def forward(self, x):
        q, k, v = self.qkv(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)).chunk(3, 1)
        if self.last:
            # The final feature map is one global window.
            self.attns[0].width = x.shape[-2]
            a = self.attns[0](q, k, v)
        else:
            a = torch.cat([att(qc, kc, vc) for att, qc, kc, vc in zip(
                self.attns, q.chunk(2, 1), k.chunk(2, 1), v.chunk(2, 1))], 1)
        x = x + self.path(self.proj(a))
        return x + self.path(self.mlp(self.norm2(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2))


def conv_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU())


class DualSegmenter(nn.Module):
    def __init__(self, pretrained=True, depths=(1, 2, 21, 1), cnn_name="convnext_small"):
        super().__init__()
        self.cnn = timm.create_model(cnn_name, pretrained=pretrained, features_only=True,
                                     in_chans=3, out_indices=(0, 1, 2, 3))
        channels = self.cnn.feature_info.channels()
        dims, heads, stripes = (32, 64, 128, 256), (2, 4, 8, 8), (1, 2, 8, 8)
        self.stem = nn.Conv2d(3, dims[0], 7, stride=4, padding=3)
        self.stages, self.down, self.align = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i, (dim, depth) in enumerate(zip(dims, depths)):
            self.stages.append(nn.Sequential(*[CSWinBlock(dim, heads[i], stripes[i], last=i == 3)
                                               for _ in range(depth)]))
            if i < 3:
                self.down.append(nn.Sequential(nn.Conv2d(dim, dims[i + 1], 3, stride=2, padding=1),
                                                nn.GroupNorm(8, dims[i + 1])))
            self.align.append(nn.Conv2d(dim, channels[i], 1))
        widths = (64, 128, 256, 256)
        self.bottleneck = conv_block(channels[3], widths[3])
        self.decoder = nn.ModuleList([conv_block(widths[i + 1] + channels[i], widths[i])
                                      for i in (2, 1, 0)])
        self.head = nn.Conv2d(widths[0], 1, 1)

    def forward(self, x):
        if x.shape[-2:] != (256, 256):
            raise ValueError("This CSWin experiment requires 256x256 slices")
        cnn = self.cnn(x)
        t, fused = self.stem(x), []
        for i, stage in enumerate(self.stages):
            t = stage(t)
            fused.append(cnn[i] + self.align[i](t))
            if i < 3:
                t = self.down[i](t)
        out = self.bottleneck(fused[3])
        for block, skip in zip(self.decoder, reversed(fused[:3])):
            out = block(torch.cat([F.interpolate(out, size=skip.shape[-2:], mode="bilinear",
                                                 align_corners=False), skip], 1))
        return F.interpolate(self.head(out), size=x.shape[-2:], mode="bilinear", align_corners=False)


def segmentation_loss(logits, target, pos_weight=500.0):
    bce = F.binary_cross_entropy_with_logits(logits, target,
                                             pos_weight=logits.new_tensor(pos_weight))
    p = logits.sigmoid().flatten(1)
    y = target.flatten(1)
    dice = (2 * (p * y).sum(1) + 1e-5) / (p.sum(1) + y.sum(1) + 1e-5)
    return bce + 0.5 * (1 - dice.mean())
