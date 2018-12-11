import torch

from ssd.modeling.post_processor import PostProcessor
from .data_preprocessing import PredictionTransform


class Predictor:
    def __init__(self, cfg, model, iou_threshold, score_threshold, device):
        self.cfg = cfg
        self.model = model
        self.transform = PredictionTransform(cfg.INPUT.IMAGE_SIZE, cfg.INPUT.PIXEL_MEAN)
        self.post_processor = PostProcessor(iou_threshold=iou_threshold,
                                            score_threshold=score_threshold,
                                            image_size=cfg.INPUT.IMAGE_SIZE,
                                            max_per_class=cfg.TEST.MAX_PER_CLASS,
                                            max_per_image=cfg.TEST.MAX_PER_IMAGE)
        self.device = device
        self.model.eval()

    def predict(self, image):
        height, width, _ = image.shape
        image = self.transform(image)
        images = image.unsqueeze(0)
        images = images.to(self.device)
        with torch.no_grad():
            scores, boxes = self.model(images)
        results = self.post_processor(scores, boxes, width=width, height=height)
        boxes, labels, scores = results[0]
        return boxes, labels, scores
