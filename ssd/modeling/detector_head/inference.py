import torch

from ssd.structures.container import Container
from ssd.utils.nms import boxes_nms


class PostProcessor:
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def __call__(self, detections):
        batches_scores, batches_boxes = detections
        device = batches_scores.device
        batch_size = batches_scores.size(0)
        results = []
        for batch_id in range(batch_size):
            processed_boxes = []
            processed_scores = []
            processed_labels = []

            scores, boxes = batches_scores[batch_id], batches_boxes[batch_id]  # (N, #CLS) (N, 4)
            for class_id in range(1, scores.size(1)):  # skip background
                per_cls_scores = scores[:, class_id]
                mask = per_cls_scores > self.cfg.TEST.CONFIDENCE_THRESHOLD
                per_cls_scores = per_cls_scores[mask]
                if per_cls_scores.numel() == 0:
                    continue
                per_cls_boxes = boxes[mask, :] * self.cfg.INPUT.IMAGE_SIZE
                keep = boxes_nms(per_cls_boxes, per_cls_scores, self.cfg.TEST.NMS_THRESHOLD, self.cfg.TEST.MAX_PER_CLASS)

                nmsed_boxes = per_cls_boxes[keep, :]
                nmsed_labels = torch.tensor([class_id] * keep.size(0), device=device)
                nmsed_scores = per_cls_scores[keep]

                processed_boxes.append(nmsed_boxes)
                processed_scores.append(nmsed_scores)
                processed_labels.append(nmsed_labels)

            if len(processed_boxes) == 0:
                processed_boxes = torch.empty(0, 4)
                processed_labels = torch.empty(0)
                processed_scores = torch.empty(0)
            else:
                processed_boxes = torch.cat(processed_boxes, 0)
                processed_labels = torch.cat(processed_labels, 0)
                processed_scores = torch.cat(processed_scores, 0)

            container = Container(boxes=processed_boxes, labels=processed_labels, scores=processed_scores)
            container.img_width = self.cfg.INPUT.IMAGE_SIZE
            container.img_height = self.cfg.INPUT.IMAGE_SIZE
            results.append(container)
        return results
