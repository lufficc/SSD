from ssd.modeling import registry
from .vgg import VGG
from .mobilenet import MobileNetV2

__all__ = ['build_backbone', 'VGG', 'MobileNetV2']


def build_backbone(cfg):
    return registry.BACKBONES[cfg.MODEL.BACKBONE.NAME](cfg, cfg.MODEL.BACKBONE.PRETRAINED)
