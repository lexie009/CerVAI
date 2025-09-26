from typing import Dict, Any
from .UNet import UNet
from .DeepLabV3Plus import DeepLabV3Plus

def build_model(name: str, cfg: Dict[str, Any]):
    name = (name or "").lower()
    if name in ["unet", "u-net"]:
        in_ch  = int(cfg.get("in_channels", 3))
        base   = int(cfg.get("base_channels", 64))
        nclass = int(cfg.get("num_classes", 2))
        return UNet(in_channels=in_ch, base_channels=base, num_classes=nclass)
    elif name in ["deeplabv3plus", "deeplabv3+", "deeplab"]:
        return DeepLabV3Plus(cfg)
    else:
        raise ValueError(f"Unknown model name: {name}")
