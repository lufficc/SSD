import torch

from ssd.structures.container import Container
from ssd.utils.nms import boxes_nms


class PostProcessor:
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.width = cfg.INPUT.IMAGE_SIZE
        self.height = cfg.INPUT.IMAGE_SIZE

    def __call__(self, detections):
        batches_scores, batches_boxes = detections
        device = batches_scores.device
        batch_size = batches_scores.size(0)
        results = []
        for batch_id in range(batch_size):
            processed_boxes = []
            processed_scores = []
            processed_labels = []

            per_img_scores, per_img_boxes = batches_scores[batch_id], batches_boxes[batch_id]  # (N, #CLS) (N, 4)
            for class_id in range(1, per_img_scores.size(1)):  # skip background
                scores = per_img_scores[:, class_id]
                mask = scores > self.cfg.TEST.CONFIDENCE_THRESHOLD
                scores = scores[mask]
                if scores.size(0) == 0:
                    continue
                boxes = per_img_boxes[mask, :]
                boxes[:, 0::2] *= self.width
                boxes[:, 1::2] *= self.height

                keep = boxes_nms(boxes, scores, self.cfg.TEST.NMS_THRESHOLD, self.cfg.TEST.MAX_PER_CLASS)

                nmsed_boxes = boxes[keep, :]
                nmsed_labels = torch.tensor([class_id] * keep.size(0), device=device)
                nmsed_scores = scores[keep]

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

            if processed_boxes.size(0) > self.cfg.TEST.MAX_PER_IMAGE > 0:
                processed_scores, keep = torch.topk(processed_scores, k=self.cfg.TEST.MAX_PER_IMAGE)
                processed_boxes = processed_boxes[keep, :]
                processed_labels = processed_labels[keep]

            container = Container(boxes=processed_boxes, labels=processed_labels, scores=processed_scores)
            container.img_width = self.width
            container.img_height = self.height
            results.append(container)
        return results
