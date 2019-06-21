import torch
from torch import nn
import torch.nn.functional as F

from ssd.modeling.anchors.prior_box import PriorBox
from ssd.utils import box_utils
from .inference import PostProcessor
from .loss import MultiBoxLoss


class SSDHeader(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cls_headers = nn.ModuleList()
        self.reg_headers = nn.ModuleList()
        for boxes_per_location, out_channels in zip(cfg.MODEL.PRIORS.BOXES_PER_LOCATION, cfg.MODEL.BACKBONE.OUT_CHANNELS):
            self.cls_headers.append(
                nn.Conv2d(out_channels, boxes_per_location * cfg.MODEL.NUM_CLASSES, kernel_size=3, stride=1, padding=1)
            )
            self.reg_headers.append(
                nn.Conv2d(out_channels, boxes_per_location * 4, kernel_size=3, stride=1, padding=1)
            )
        self.loss_evaluator = MultiBoxLoss(neg_pos_ratio=cfg.MODEL.NEG_POS_RATIO)
        self.post_processor = PostProcessor(cfg)
        self.priors = None
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features, targets=None):
        confidences = []
        locations = []
        for feature, cls_header, reg_header in zip(features, self.cls_headers, self.reg_headers):
            confidences.append(cls_header(feature).permute(0, 2, 3, 1).contiguous())
            locations.append(reg_header(feature).permute(0, 2, 3, 1).contiguous())

        batch_size = features[0].shape[0]
        confidences = torch.cat([c.view(c.shape[0], -1) for c in confidences], dim=1).view(batch_size, -1, self.cfg.MODEL.NUM_CLASSES)
        locations = torch.cat([l.view(l.shape[0], -1) for l in locations], dim=1).view(batch_size, -1, 4)

        if self.training:
            return self._forward_train(confidences, locations, targets)
        else:
            return self._forward_test(confidences, locations)

    def _forward_train(self, confidences, locations, targets):
        gt_boxes, gt_labels = targets['boxes'], targets['labels']
        reg_loss, cls_loss = self.loss_evaluator(confidences, locations, gt_labels, gt_boxes)
        loss_dict = dict(
            reg_loss=reg_loss,
            cls_loss=cls_loss,
        )
        detections = (confidences, locations)
        return detections, loss_dict

    def _forward_test(self, confidences, locations):
        if self.priors is None:
            self.priors = PriorBox(self.cfg)().to(locations.device)
        scores = F.softmax(confidences, dim=2)
        boxes = box_utils.convert_locations_to_boxes(
            locations, self.priors, self.cfg.MODEL.CENTER_VARIANCE, self.cfg.MODEL.SIZE_VARIANCE
        )
        boxes = box_utils.center_form_to_corner_form(boxes)
        detections = (scores, boxes)
        detections = self.post_processor(detections)
        return detections, {}
