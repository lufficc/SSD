from ssd.data.datasets import VOCDataset, COCODataset
from .coco import coco_evaluation
from .voc import voc_evaluation


def evaluate(dataset, predictions, output_dir):
    args = dict(
        dataset=dataset, predictions=predictions, output_dir=output_dir
    )
    if isinstance(dataset, VOCDataset):
        return voc_evaluation(**args)
    elif isinstance(dataset, COCODataset):
        return coco_evaluation(**args)
