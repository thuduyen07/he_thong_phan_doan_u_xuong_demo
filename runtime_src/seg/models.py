from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import SegformerConfig, SegformerForSemanticSegmentation


@dataclass
class ModelSpec:
    model_type: str
    checkpoint_kind: str


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.shape[-2] - x.shape[-2]
        diff_x = skip.shape[-1] - x.shape[-1]
        if diff_y != 0 or diff_x != 0:
            x = torch.nn.functional.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        return self.conv(torch.cat([skip, x], dim=1))


class LightweightUNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 2, base_channels: int = 16) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.inc = DoubleConv(in_channels, c1)
        self.down1 = DownBlock(c1, c2)
        self.down2 = DownBlock(c2, c3)
        self.down3 = DownBlock(c3, c4)
        self.down4 = DownBlock(c4, c5)
        self.up1 = UpBlock(c5, c4, c4)
        self.up2 = UpBlock(c4, c3, c3)
        self.up3 = UpBlock(c3, c2, c2)
        self.up4 = UpBlock(c2, c1, c1)
        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.head(x)


def get_model_type(cfg: dict) -> str:
    return str(cfg.get("model_type", "segformer")).strip().lower()


def build_segformer(cfg: dict):
    model_name = cfg.get("local_model_path") or cfg["model_name"]
    pretrained = cfg.get("pretrained", True)
    num_classes = cfg["num_classes"]
    local_files_only = bool(cfg.get("local_files_only", False))

    if pretrained:
        try:
            return SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            print(
                f"Warning: failed to load pretrained SegFormer weights from `{model_name}`. "
                f"Falling back to randomly initialized SegFormer. Root cause: {exc}"
            )

    seg_cfg = SegformerConfig(num_labels=num_classes, num_channels=3)
    return SegformerForSemanticSegmentation(seg_cfg)


def build_unet(cfg: dict) -> nn.Module:
    unet_cfg = dict(cfg.get("unet", {}))
    return LightweightUNet(
        in_channels=int(unet_cfg.get("in_channels", 3)),
        num_classes=int(cfg["num_classes"]),
        base_channels=int(unet_cfg.get("base_channels", 16)),
    )


def build_segmentation_model(cfg: dict) -> tuple[nn.Module, ModelSpec]:
    model_type = get_model_type(cfg)
    if model_type == "segformer":
        return build_segformer(cfg), ModelSpec(model_type=model_type, checkpoint_kind="segformer")
    if model_type == "unet":
        return build_unet(cfg), ModelSpec(model_type=model_type, checkpoint_kind="plain")
    raise ValueError(f"Unsupported model_type: {model_type}")


def extract_logits(model, x: torch.Tensor, model_type: str) -> torch.Tensor:
    if model_type == "segformer":
        return model(pixel_values=x).logits
    if model_type == "unet":
        return model(x)
    raise ValueError(f"Unsupported model_type: {model_type}")
