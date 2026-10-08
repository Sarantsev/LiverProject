"""Merlin (Stanford MIMI, Nature 2026) abdominal-CT encoder as a classification backbone.

Vendored, dependency-free port of the IMAGE tower of ``merlin-vlm`` (MIT): an I3D-inflated
torchvision ResNet-152 (``merlin/models/i3res.py`` + ``inflate.py``). The PyPI package pins
torch>=2.1 / monai>=1.3 / transformers>=4.38, which cannot coexist with the SegVol stack
(torch 1.13 / monai 0.9), and it would also pull the Clinical-Longformer text tower we do
not use. Weights are fetched from HF ``stanfordmimi/Merlin`` and loaded into the image
tower with a strict key check.

Pretraining recipe reproduced via ``preprocess_spec``: 1.5 mm in-plane / 3 mm slices,
HU [-1000, 1000] -> [0, 1], a 224x224x160 window (zero pad / crop). Merlin's own
linear-probe recipe centres that window on a foreground mask -- exactly what roi=pred does.

Layout: Merlin feeds (B,1,H,W,D) and permutes to (B,1,D,H,W) inside forward; this port
takes (B,1,D,H,W) directly. Output = the layer4 feature map passed through the pretrained
CLIP ``contrastive_head`` (1x1x1 conv 2048->512). Because that head is linear and global
pooling is linear, projecting before pooling equals Merlin's pooled 512-d embedding, while
keeping a spatial map for masked pooling -> (B, 512, D/16, H/32, W/32).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from .backbones import Backbone

HF_REPO = "stanfordmimi/Merlin"
HF_CKPT = "i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt"
_PREFIX = "encode_image.i3_resnet."


# ---------------------------- inflation (2D ResNet -> 3D) ----------------------------
def _inflate_conv(c2, time_dim=3, time_padding=0, time_stride=1):
    if c2.kernel_size[0] == 7:                                  # stem: (3,7,7), stride (1,2,2)
        k, p, s = (3, 7, 7), (1, 3, 3), (1, 2, 2)
    else:
        k = (time_dim, c2.kernel_size[0], c2.kernel_size[1])
        p = (time_padding, c2.padding[0], c2.padding[1])
        s = (time_stride, c2.stride[0], c2.stride[0])
    return nn.Conv3d(c2.in_channels, c2.out_channels, k, stride=s, padding=p, bias=c2.bias is not None)


def _inflate_bn(bn2):
    return nn.BatchNorm3d(bn2.num_features, eps=bn2.eps, momentum=bn2.momentum)


class _Bottleneck3d(nn.Module):
    def __init__(self, b2):
        super().__init__()
        s = b2.conv2.stride[0]
        self.conv1 = _inflate_conv(b2.conv1, time_dim=1); self.bn1 = _inflate_bn(b2.bn1)
        self.conv2 = _inflate_conv(b2.conv2, time_dim=3, time_padding=1, time_stride=s); self.bn2 = _inflate_bn(b2.bn2)
        self.conv3 = _inflate_conv(b2.conv3, time_dim=1); self.bn3 = _inflate_bn(b2.bn3)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if b2.downsample is not None:
            self.downsample = nn.Sequential(_inflate_conv(b2.downsample[0], time_dim=1, time_stride=s),
                                            _inflate_bn(b2.downsample[1]))

    def _body(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return self.bn3(self.conv3(out))

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = checkpoint.checkpoint(self._body, x) if x.requires_grad else self._body(x)
        return self.relu(out + residual)


class I3ResNetEncoder(nn.Module):
    """conv1 .. layer4 of Merlin's I3ResNet (+ the CLIP contrastive_head). No text tower,
    no phenotype classifier. features: 'contrastive' (512-d) | 'layer4' (2048-d)."""
    def __init__(self, resnet2d, features: str = "contrastive"):
        super().__init__()
        if features not in ("contrastive", "layer4"):
            raise ValueError("features must be 'contrastive' or 'layer4'")
        self.features = features
        self.conv1 = _inflate_conv(resnet2d.conv1)
        self.bn1 = _inflate_bn(resnet2d.bn1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.layer1 = nn.Sequential(*[_Bottleneck3d(b) for b in resnet2d.layer1])
        self.layer2 = nn.Sequential(*[_Bottleneck3d(b) for b in resnet2d.layer2])
        self.layer3 = nn.Sequential(*[_Bottleneck3d(b) for b in resnet2d.layer3])
        self.layer4 = nn.Sequential(*[_Bottleneck3d(b) for b in resnet2d.layer4])
        self.contrastive_head = nn.Conv3d(2048, 512, kernel_size=1, bias=True)
        self.out_dim = 512 if features == "contrastive" else 2048

    def forward(self, x):                                       # (B,1,D,H,W)
        x = torch.cat((x, x, x), dim=1)                         # ImageNet-inflated 3-channel stem
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            x = checkpoint.checkpoint(layer, x) if (self.training and x.requires_grad) else layer(x)
        return self.contrastive_head(x) if self.features == "contrastive" else x


# ---------------------------------- backbone ----------------------------------
class MerlinBackbone(Backbone):
    def __init__(self, encoder: I3ResNetEncoder, spatial_size: Sequence[int] = (160, 224, 224),
                 spacing: Sequence[float] = (3.0, 1.5, 1.5), hu_window: Sequence[float] = (-1000, 1000),
                 freeze_bn: bool = True):
        super().__init__()
        self.i3_resnet = encoder
        self.embed_dim = encoder.out_dim
        self.freeze_bn = freeze_bn
        self.preprocess_spec = {"spatial_size": tuple(spatial_size), "spacing": tuple(spacing),
                                "normalize": "hu", "hu_window": tuple(hu_window)}

    @property
    def encoder(self) -> nn.Module:
        return self.i3_resnet

    def train(self, mode: bool = True):
        """Keep BatchNorm in eval mode while finetuning (freeze_bn, the default).

        This backbone is a ResNet. In train mode every BatchNorm3d normalises by the
        statistics of the current batch AND overwrites its pretrained running_mean /
        running_var -- those are buffers, so requires_grad=False does NOT protect them and
        even the frozen stages are affected. Our batch is 2 split over 2 GPUs, i.e. ONE
        volume per BatchNorm call, so those statistics are pure noise and the pretrained
        features are destroyed within the first epoch: the classifier then collapses onto a
        single class (observed: HH sens 1.00 / spec 0.39, every other class sens ~0.1,
        while the AUCs stayed high -- ranking survived, decisions did not).

        Freezing BN is the standard way to finetune a convolutional backbone at small batch
        size. SegVol needs none of this: it is a ViT with LayerNorm, which is
        batch-independent -- which is why only this backbone collapsed.
        """
        super().train(mode)
        if mode and self.freeze_bn:
            n = 0
            for m in self.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
                    n += 1
            if not getattr(self, "_bn_frozen_logged", False):
                print(f"Merlin: BatchNorm frozen in eval mode ({n} layers) -- pretrained "
                      f"running stats preserved")
                self._bn_frozen_logged = True
        return self

    def forward(self, x):
        return self.i3_resnet(x)

    @classmethod
    def from_pretrained(cls, m_cfg: dict, cache_dir: str | None = None) -> "MerlinBackbone":
        """Build the inflated ResNet-152 skeleton and load Merlin's image-tower weights."""
        import torchvision
        from huggingface_hub import hf_hub_download
        enc = I3ResNetEncoder(torchvision.models.resnet152(weights=None),
                              features=m_cfg.get("features", "contrastive"))
        path = m_cfg.get("checkpoint") or hf_hub_download(repo_id=HF_REPO, filename=HF_CKPT,
                                                          cache_dir=cache_dir)
        sd = torch.load(path, map_location="cpu")
        sub = {k[len(_PREFIX):]: v for k, v in sd.items()
               if k.startswith(_PREFIX) and not k.startswith(_PREFIX + "classifier.")}
        missing, unexpected = enc.load_state_dict(sub, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Merlin image tower mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
        print(f"Merlin: loaded {len(sub)} image-tower tensors from {path}")
        return cls(enc, spatial_size=m_cfg["spatial_size"], spacing=m_cfg["spacing"],
                   hu_window=m_cfg.get("hu_window", (-1000, 1000)),
                   freeze_bn=m_cfg.get("freeze_bn", True))
