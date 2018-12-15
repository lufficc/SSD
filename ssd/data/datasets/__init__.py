from torch.utils.data import ConcatDataset

from ssd.config.path_catlog import DatasetCatalog
from .voc_dataset import VOCDataset
from .coco_dataset import COCODataset

_DATASETS = {
    'VOCDataset': VOCDataset,
    'COCODataset': COCODataset,
}


def build_dataset(dataset_list, transform=None, target_transform=None, is_test=False):
    assert len(dataset_list) > 0
    datasets = []
    for dataset_name in dataset_list:
        data = DatasetCatalog.get(dataset_name)
        args = data['args']
        factory = _DATASETS[data['factory']]
        args['transform'] = transform
        args['target_transform'] = target_transform
        if factory == VOCDataset:
            args['keep_difficult'] = is_test
        elif factory == COCODataset:
            args['remove_empty'] = not is_test
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
