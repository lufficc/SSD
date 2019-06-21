from .vgg import vgg

BACKBONES = {
    'vgg': vgg,
}


def build_backbone(cfg):
    return BACKBONES[cfg.MODEL.BACKBONE.NAME](cfg)
