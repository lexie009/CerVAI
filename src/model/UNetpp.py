import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.nn.functional as F
import torch


class UNetpp(nn.Module):
    """
    UNet++ implemented via segmentation_models_pytorch.
    Interface is aligned with your DeepLabV3Plus:
      - forward(x, mc_dropout=False) -> logits (no sigmoid/softmax)
      - sample_predict(x, T=10) -> [T, B, C, H, W]
    """
    def __init__(self, model_cfg):
        super().__init__()

        structure = model_cfg.get("structure", {})
        encoder_cfg = structure.get("encoder", {})
        dropout_cfg = structure.get("dropout", {})

        in_channels = int(model_cfg.get("in_channels", 3))

        # Keep output channels consistent with DeepLabV3Plus config style:
        # use out_channels if provided, otherwise fallback to 1 for binary.
        out_channels = int(model_cfg.get("out_channels", model_cfg.get("num_classes", 1)))

        encoder_name = encoder_cfg.get("name", "resnet34")
        encoder_weights = encoder_cfg.get("encoder_weights", "imagenet")

        # Optional dropout (MC dropout friendly)
        self.use_dropout = bool(dropout_cfg.get("enable", False))
        self.dropout_p = float(dropout_cfg.get("dropout_rate", 0.5))

        # Build UNet++ (nested skip connections)
        self.base_model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
        )

        # Match your DeepLabV3Plus behavior: output logits only
        self.activation = nn.Identity()

    def forward(self, x, mc_dropout: bool = False):
        logits = self.base_model(x)  # raw logits, shape [B, C, H, W]

        # Optional dropout for MC sampling (apply on logits like your DeepLabV3Plus)
        if self.use_dropout and mc_dropout:
            logits = F.dropout2d(logits, p=self.dropout_p, training=True)

        return logits

    def sample_predict(self, x, T: int = 10):
        self.eval()
        preds = []
        for _ in range(T):
            with torch.no_grad():
                preds.append(self.forward(x, mc_dropout=True).unsqueeze(0))
        return torch.cat(preds, dim=0)  # [T, B, C, H, W]
