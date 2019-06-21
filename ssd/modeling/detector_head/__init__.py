from .detector_head import SSDHeader


def build_detector_head(cfg):
    # TODO: make it more general
    return SSDHeader(cfg)
