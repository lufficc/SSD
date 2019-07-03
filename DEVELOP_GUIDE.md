# Develop Guide

## Custom Dataset
Add your custom dataset is simple and flexible.
For example, create `ssd/data/datasets/my_dataset.py`:
```python
import torch.utils.data

from ssd.structures.container import Container

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, ..., transform=None, target_transform=None):
        # as you would do normally
        ...
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        # load the image as a PIL Image
        image = ...

        # load the bounding boxes in x1, y1, x2, y2 order.
        boxes = np.array((N, 4), dtype=np.float32)
        # and labels
        labels = np.array((N, ), dtype=np.int64)

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        targets = Container(
            boxes=boxes,
            labels=labels,
        )
        # return the image, the targets and the index in your dataset
        return image, targets, index
```

in `ssd/data/datasets/__init__.py`
```python
from .my_dataset import MyDataset

_DATASETS = {
    'VOCDataset': VOCDataset,
    'COCODataset': COCODataset,
    'MyDataset': MyDataset,
}
```

in `ssd/config/path_catlog.py`:
```python
DATASETS = {
    ...
    'my_custom_dataset': {
        "arg1": "your/arg",
        "arg2": "your/arg",
    },
    ...
}

@staticmethod
def get(name):
    ...
    if name == 'my_custom_dataset':
        attrs = DatasetCatalog.DATASETS[name]
        return dict(factory="MyDataset", args=attrs)
    ...
```

in your `config.ymal`:
```yaml
DATASETS:
  TRAIN: ("my_custom_dataset", )
  TEST: ("my_custom_test_dataset", )
```

### Test
While the aforementioned example should work for training, it's also easy to add your custom test code:
in `ssd/data/datasets/evaluation/__init__.py`
```python
if isinstance(dataset, MyDataset):
    return my_own_evaluation(**args)
```

## Custom Backbone

It very simple to add your own backbone for SSD.
For example, create `ssd/modeling/backbone/my_backbone.py`:
```python
import torch.nn as nn

from ssd.modeling import registry
from ssd.utils.model_zoo import load_state_dict_from_url


class MyBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ...

    def forward(self, x):
        features = []
        
        # forward your network
        
        # add arbitrary feature you want to do prediction upon it.
        
        features.append(feature1)
        features.append(feature2)
        features.append(feature3)
        features.append(feature4)

        # return them as a tuple
        return tuple(features)

@registry.BACKBONES.register('my_backbone')
def my_backbone(cfg, pretrained=True):
    model = MyBackbone(cfg)
    model_url = 'you_model_url'
    if pretrained:
        model.init_from_pretrain(load_state_dict_from_url(model_url))
    return model
```
in `ssd/modeling/backbone/__init__.py`:
```python
from .my_backbone import MyBackbone
```

in your `config.ymal`:
```yaml
MODEL:
  BACKBONE:
    NAME: 'my_backbone'
    OUT_CHANNELS: (-, -, -, -) # should match feature1 - feature4's out_channels in MyBackbone
  PRIORS:
    FEATURE_MAPS: [-, -, -, -] # feature1 - feature4's size
    STRIDES: [-, -, -, -] # feature1 - feature4's output stride
    MIN_SIZES: [21, 45, 99, 153] # your custom anchor settings
    MAX_SIZES: [45, 99, 153, 207]
    ASPECT_RATIOS: [[2, 3], [2, 3], [2, 3], [2, 3]]
    BOXES_PER_LOCATION: [6, 6, 6, 6]
```