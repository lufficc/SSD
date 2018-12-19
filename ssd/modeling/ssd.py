import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

from ssd.modeling.multibox_loss import MultiBoxLoss
from ssd.module import L2Norm
from ssd.module.prior_box import PriorBox
from ssd.utils import box_utils


class SSD(nn.Module):
    def __init__(self, cfg,
                 vgg: nn.ModuleList,
                 extras: nn.ModuleList,
                 classification_headers: nn.ModuleList,
                 regression_headers: nn.ModuleList):
        """Compose a SSD model using the given components.
        """
        super(SSD, self).__init__()
        self.cfg = cfg
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.vgg = vgg
        self.extras = extras
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.l2_norm = L2Norm(512, scale=20)
        self.criterion = MultiBoxLoss(neg_pos_ratio=cfg.MODEL.NEG_POS_RATIO)
        self.priors = None
        self.reset_parameters()

    def reset_parameters(self):
        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        self.vgg.apply(weights_init)
        self.extras.apply(weights_init)
        self.classification_headers.apply(weights_init)
        self.regression_headers.apply(weights_init)

    def forward(self, x, targets=None):
        sources = []
        confidences = []
        locations = []
        for i in range(23):
            x = self.vgg[i](x)
        s = self.l2_norm(x)  # Conv4_3 L2 normalization
        sources.append(s)

        # apply vgg up to fc7
        for i in range(23, len(self.vgg)):
            x = self.vgg[i](x)
        sources.append(x)

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        for (x, l, c) in zip(sources, self.regression_headers, self.classification_headers):
            locations.append(l(x).permute(0, 2, 3, 1).contiguous())
            confidences.append(c(x).permute(0, 2, 3, 1).contiguous())

        confidences = torch.cat([o.view(o.size(0), -1) for o in confidences], 1)
        locations = torch.cat([o.view(o.size(0), -1) for o in locations], 1)

        confidences = confidences.view(confidences.size(0), -1, self.num_classes)
        locations = locations.view(locations.size(0), -1, 4)

        if not self.training:
            # when evaluating, decode predictions
            if self.priors is None:
                self.priors = PriorBox(self.cfg)().to(locations.device)
            confidences = F.softmax(confidences, dim=2)
            boxes = box_utils.convert_locations_to_boxes(
                locations, self.priors, self.cfg.MODEL.CENTER_VARIANCE, self.cfg.MODEL.SIZE_VARIANCE
            )
            boxes = box_utils.center_form_to_corner_form(boxes)
            return confidences, boxes
        else:
            # when training, compute losses
            gt_boxes, gt_labels = targets
            regression_loss, classification_loss = self.criterion(confidences, locations, gt_labels, gt_boxes)
            loss_dict = dict(
                regression_loss=regression_loss,
                classification_loss=classification_loss,
            )
            return loss_dict

    def init_from_base_net(self, model):
        vgg_weights = torch.load(model, map_location=lambda storage, loc: storage)
        self.vgg.load_state_dict(vgg_weights, strict=True)

    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)


class MatchPrior(object):
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
        self.center_form_priors = center_form_priors
        self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes, labels = box_utils.assign_priors(gt_boxes, gt_labels,
                                                self.corner_form_priors, self.iou_threshold)
        boxes = box_utils.corner_form_to_center_form(boxes)
        locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance, self.size_variance)
        return locations, labels
