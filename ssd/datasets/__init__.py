from torch.utils.data import ConcatDataset

from ssd.config.path_catlog import DatasetCatalog
from .voc_dataset import VOCDataset


def build_dataset(dataset_list, transform=None, target_transform=None):
    assert len(dataset_list) > 0
    datasets = []
    for dataset_name in dataset_list:
        data = DatasetCatalog.get(dataset_name)
        args = data['args']
        factory = globals()[data['factory']]
        if data['factory'] == 'VOCDataset':
            args['target_transform'] = target_transform
        args['transform'] = transform
        dataset = factory(**args)
        datasets.append(dataset)
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)
    else:
        dataset = datasets[0]

    return dataset
