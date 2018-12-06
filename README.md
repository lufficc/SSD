# High quality, fast, modular reference implementation of SSD in PyTorch 1.0


This repository implements [SSD (Single Shot MultiBox Detector)](https://arxiv.org/abs/1512.02325). The implementation is heavily influenced by the projects [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch), [pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd) and [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). This repository aims to be the code base for researches based on SSD.

## Installation
### Requirements
1. Python3
1. PyTorch 1.0
1. yacs
1. GCC >= 4.9
1. OpenCV
### Build
```
# build nms
cd ext
python build.py build_ext develop
```

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
| SSD512* |      -       |

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
<td><pre><code>-</code></pre></td>
</tr>
</tbody></table>
