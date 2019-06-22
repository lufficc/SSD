from ssd.modeling import registry
from ssd.utils.model_zoo import load_state_dict_from_url
from .vgg import VGG
from .vgg import model_urls as vgg_model_urls
from .mobilenet import MobileNetV2
from .mobilenet import model_urls as mobilenet_model_urls


@registry.BACKBONES.register('vgg')
def vgg(cfg, pretrained=True):
    model = VGG(cfg)
    if pretrained:
        model.init_from_pretrain(load_state_dict_from_url(vgg_model_urls['vgg']))
    return model


@registry.BACKBONES.register('mobilenet_v2')
def mobilenet_v2(cfg, pretrained=False):
    model = MobileNetV2()
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(mobilenet_model_urls['mobilenet_v2']))
    return model


def build_backbone(cfg):
    return registry.BACKBONES[cfg.MODEL.BACKBONE.NAME](cfg)
