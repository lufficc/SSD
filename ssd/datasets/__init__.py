from torch.utils.data import ConcatDataset

from ssd.config.path_catlog import DatasetCatalog
from .voc_dataset import VOCDataset
from .coco_dataset import COCODataset


def build_dataset(dataset_list, transform=None, target_transform=None, is_test=False):
    assert len(dataset_list) > 0
    datasets = []
    for dataset_name in dataset_list:
        data = DatasetCatalog.get(dataset_name)
        args = data['args']
        factory = globals()[data['factory']]
        args['transform'] = transform
        args['target_transform'] = target_transform
        dataset = factory(**args)
        datasets.append(dataset)
    # for testing, return a list of datasets
    if is_test:
        return datasets
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)
    else:
        dataset = datasets[0]

    return dataset
