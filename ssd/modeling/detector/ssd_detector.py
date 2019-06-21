from torch import nn

from ssd.modeling.backbone import build_backbone
from ssd.modeling.detector_head import build_detector_head


class SSDDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.detector_head = build_detector_head(cfg)

    def forward(self, images, targets=None):
        features = self.backbone(images)
        detections, detector_losses = self.detector_head(features, targets)
        if self.training:
            return detector_losses
        return detections
