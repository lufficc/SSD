import torch.nn as nn
import torch.nn.functional as F
import torch

from ssd.utils import box_utils

class IouLoss(nn.Module):

    def __init__(self,pred_mode = 'Center',size_sum=True,variances=None,losstype='Giou'):
        super(IouLoss, self).__init__()
        self.size_sum = size_sum
        self.pred_mode = pred_mode
        self.variances = variances
        self.loss = losstype
    def forward(self, loc_p, loc_t,prior_data):
        num = loc_p.shape[0] 
        
        if self.pred_mode == 'Center':
            decoded_boxes = box_utils.decode(loc_p, prior_data, self.variances)
        else:
            decoded_boxes = loc_p
        if self.loss == 'Iou':
            loss = torch.sum(1.0 - box_utils.bbox_overlaps_iou(decoded_boxes, loc_t))
        else:
            if self.loss == 'Giou':
                loss = torch.sum(1.0 - box_utils.bbox_overlaps_giou(decoded_boxes,loc_t))
            else:
                if self.loss == 'Diou':
                    loss = torch.sum(1.0 - box_utils.bbox_overlaps_diou(decoded_boxes,loc_t))
                else:
                    loss = torch.sum(1.0 - box_utils.bbox_overlaps_ciou(decoded_boxes, loc_t))            
     
        if self.size_sum:
            loss = loss
        else:
            loss = loss/num
        return loss

class MultiBoxLoss(nn.Module):
    def __init__(self, neg_pos_ratio, losstype):
        """Implement SSD MultiBox Loss.

        Basically, MultiBox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiBoxLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.losstype = losstype
        self.gious = IouLoss(pred_mode = 'Center',size_sum=True,variances=self.variance, losstype=losstype)

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            predicted_locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            gt_locations (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        priors = predicted_locations[1:]
        priors = priors[:confidence.size(1), :]
        
        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)

        confidence = confidence[mask, :]
        classification_loss = F.cross_entropy(confidence.view(-1, num_classes), labels[mask], reduction='sum')

        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].view(-1, 4)
        gt_locations = gt_locations[pos_mask, :].view(-1, 4)
        
        if self.losstype == 'SmoothL1':
            smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction='sum')
        else:
            giou_priors = priors.data.unsqueeze(0).expand_as(predicted_locations)
            smooth_l1_loss = self.gious(predicted_locations, gt_locations, giou_priors[pos_idx].view(-1, 4))

        num_pos = gt_locations.size(0)
        return smooth_l1_loss / num_pos, classification_loss / num_pos
