from .ssd_detector import SSDDetector

_DETECTION_META_ARCHITECTURES = {
    "SSDDetector": SSDDetector
}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
