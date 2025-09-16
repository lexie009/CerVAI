import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.nn.functional as F
import torch

class DeepLabV3Plus(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        structure = model_cfg["structure"]
        encoder_name = structure["encoder"].get("name", "resnet50")
        encoder_weights = structure["encoder"].get("encoder_weights", "imagenet")
        output_stride = structure["encoder"].get("output_stride", 16)
        in_channels = model_cfg.get("in_channels", 3)
        out_channels = model_cfg.get("out_channels", 1)

        # === Create backbone + decoder ===
        self.base_model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
            decoder_atrous_rates=structure["decoder"].get("atrous_rates", [6, 12, 18])
        )

        # === Optional Dropout ===
        dropout_cfg = structure.get("dropout", {})
        self.use_dropout = dropout_cfg.get("enable", False)
        self.dropout_p = dropout_cfg.get("dropout_rate", 0.5)

        # === Output activation ===
        head_cfg = structure.get("head", {})
        activation = head_cfg.get("activation", "sigmoid")
        if activation == "softmax":
            self.activation = nn.Softmax(dim=1)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Identity()

    def forward(self, x, mc_dropout=False):
        masks = self.base_model(x)

        if self.use_dropout and mc_dropout:
            masks = F.dropout2d(masks, p=self.dropout_p, training=True)

        return self.activation(masks)

    def sample_predict(self, x, T=10):
        self.eval()
        preds = []
        for _ in range(T):
            with torch.no_grad():
                preds.append(self.forward(x, mc_dropout=True).unsqueeze(0))
        return torch.cat(preds, dim=0)  # [T, B, C, H, W]


