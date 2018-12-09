# High quality, fast, modular reference implementation of SSD in PyTorch 1.0


This repository implements [SSD (Single Shot MultiBox Detector)](https://arxiv.org/abs/1512.02325). The implementation is heavily influenced by the projects [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch), [pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd) and [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). This repository aims to be the code base for researches based on SSD.

## Highlights
- PyTorch 1.0
- GPU/CPU NMS
- Multi-GPU training and inference
- Modular
- Visualization(Support Tensorboard)
- CPU support for inference

## Installation
### Requirements
1. Python3
1. PyTorch 1.0
1. yacs
1. GCC >= 4.9
1. OpenCV
### Build
```bash
# build nms
cd ext
python build.py build_ext develop
```

## Train

### Setting Up Datasets
#### Pascal VOC
For Pascal VOC dataset, make the folder structure like this:
```
VOC_ROOT
|__ VOC2007
    |_ JPEGImages
    |_ Annotations
    |_ ImageSets
    |_ SegmentationClass
|__ VOC2012
    |_ JPEGImages
    |_ Annotations
    |_ ImageSets
    |_ SegmentationClass
|__ ...
```
Where `VOC_ROOT` default is `datasets` folder in current project, you can create symlinks to `datasets` or `export VOC_ROOT="/path/to/voc_root"`.
#### COCO
For COCO dataset, make the folder structure like this:
```
COCO_ROOT
|__ annotations
    |_ instances_valminusminival2014.json
    |_ instances_minival2014.json
    |_ instances_train2014.json
    |_ instances_val2014.json
    |_ ...
|__ train2014
    |_ <im-1-name>.jpg
    |_ ...
    |_ <im-N-name>.jpg
|__ val2014
    |_ <im-1-name>.jpg
    |_ ...
    |_ <im-N-name>.jpg
|__ ...
```
Where `COCO_ROOT` default is `datasets` folder in current project, you can create symlinks to `datasets` or `export COCO_ROOT="/path/to/coco_root"`.

### Single GPU training
```bash
# for example, train SSD300:
python train_ssd.py --config-file configs/ssd300_voc0712.yaml --vgg vgg16_reducedfc.pth
```
### Multi-GPU training
```bash
# for example, train SSD300 with 4 GPUs:
export NGPUS=4
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_ssd.py --config-file configs/ssd300_voc0712.yaml --vgg vgg16_reducedfc.pth
```
The configuration files that I provide assume that we are running on single GPU. When changing number of GPUs, hyper-parameter (lr, max_iter, ...) will also changed according to this paper: [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677).
The pre-trained vgg weights can be downloaded here: https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth.

## Demo
Predicting image in a folder is simple:
```bash
python demo.py --config-file configs/ssd300_voc0712.yaml --weights path/to/trained/weights.pth --images_dir demo
```
Then the predicted images with boxes, scores and label names will saved to `demo/result` folder.

Currently, I provide weights trained as follows:

|         |    Weights   |
| :-----: | :----------: |
| SSD300* | [ssd300_voc0712_mAP77.83.pth(100 MB)](https://github.com/lufficc/SSD/releases/download/v1.0/ssd300_voc0712_mAP77.83.pth) |
| SSD512* | [ssd512_voc0712_mAP80.25.pth(104 MB)](https://github.com/lufficc/SSD/releases/download/v1.0/ssd512_voc0712_mAP80.25.pth) |

## Performance
### Origin Paper:

|         | VOC2007 test |
| :-----: | :----------: |
| SSD300* |     77.2     |
| SSD512* |     79.8     |

### Our Implementation:

|         | VOC2007 test |
| :-----: | :----------: |
| SSD300* |     77.8     |
| SSD512* |     80.2     |

### Details:

<table>
<thead>
<tr>
<th></th>
<th>VOC2007 test</th>
</tr>
</thead>
<tbody>
<tr>
<td>SSD300*</td>
<td><pre><code>mAP: 0.7783
aeroplane       : 0.8252
bicycle         : 0.8445
bird            : 0.7597
boat            : 0.7102
bottle          : 0.5275
bus             : 0.8643
car             : 0.8660
cat             : 0.8741
chair           : 0.6179
cow             : 0.8279
diningtable     : 0.7862
dog             : 0.8519
horse           : 0.8630
motorbike       : 0.8515
person          : 0.8024
pottedplant     : 0.5079
sheep           : 0.7685
sofa            : 0.7926
train           : 0.8704
tvmonitor       : 0.7554</code></pre></td>
</tr>
<tr>
<td>SSD512*</td>
<td><pre><code>mAP: 0.8025
aeroplane       : 0.8582
bicycle         : 0.8710
bird            : 0.8192
boat            : 0.7410
bottle          : 0.5894
bus             : 0.8755
car             : 0.8856
cat             : 0.8926
chair           : 0.6589
cow             : 0.8634
diningtable     : 0.7676
dog             : 0.8707
horse           : 0.8806
motorbike       : 0.8512
person          : 0.8316
pottedplant     : 0.5238
sheep           : 0.8191
sofa            : 0.7915
train           : 0.8735
tvmonitor       : 0.7866</code></pre></td>
</tr>
</tbody></table>
