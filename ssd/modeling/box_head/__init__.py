from .box_head import SSDBoxHead


def build_box_head(cfg):
    # TODO: make it more general
    return SSDBoxHead(cfg)
