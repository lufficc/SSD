import torch.nn as nn
from ssd.modeling.ssd import SSD


# borrowed from https://github.com/amdegroot/ssd.pytorch/blob/master/ssd.py
def add_vgg(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, size=300):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    if size == 512:
        layers.append(nn.Conv2d(in_channels, 128, kernel_size=1, stride=1))
        layers.append(nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1))
    return layers


def add_header(vgg, extra_layers, boxes_per_location, num_classes):
    regression_headers = []
    classification_headers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        regression_headers += [nn.Conv2d(vgg[v].out_channels,
                                         boxes_per_location[k] * 4, kernel_size=3, padding=1)]
        classification_headers += [nn.Conv2d(vgg[v].out_channels,
                                             boxes_per_location[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        regression_headers += [nn.Conv2d(v.out_channels, boxes_per_location[k]
                                         * 4, kernel_size=3, padding=1)]
        classification_headers += [nn.Conv2d(v.out_channels, boxes_per_location[k]
                                             * num_classes, kernel_size=3, padding=1)]
    return regression_headers, classification_headers


def build_ssd_model(cfg):
    num_classes = cfg.MODEL.NUM_CLASSES
    size = cfg.INPUT.IMAGE_SIZE
    vgg_base = {
        '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
                512, 512, 512],
        '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
                512, 512, 512],
    }
    extras_base = {
        '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
        '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
    }

    boxes_per_location = cfg.MODEL.PRIORS.BOXES_PER_LOCATION

    vgg_config = vgg_base[str(size)]
    extras_config = extras_base[str(size)]

    vgg = nn.ModuleList(add_vgg(vgg_config))
    extras = nn.ModuleList(add_extras(extras_config, i=1024, size=size))

    regression_headers, classification_headers = add_header(vgg, extras, boxes_per_location, num_classes=num_classes)
    regression_headers = nn.ModuleList(regression_headers)
    classification_headers = nn.ModuleList(classification_headers)

    return SSD(cfg=cfg,
               vgg=vgg,
               extras=extras,
               classification_headers=classification_headers,
               regression_headers=regression_headers)
