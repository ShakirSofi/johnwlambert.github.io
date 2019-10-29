---
layout: post
title:  "Modern Semantic Segmentation"
permalink: /sem-seg/
excerpt: "IoU, PSPNet, ADE20K, COCO, Mapillary Vistas, CamVid ..."
mathjax: true
date:   2018-12-27 11:00:00
mathjax: true

---
Table of Contents:
- [What is Semantic Segmentation?](#state-estimation)
- [How do we evaluate this task?](#dt-lds)
- [PSPNet](#probability-review)

<a name='state-estimation'></a>

## What is Semantic Segmentation?

The goal is to assign each pixel in the image a category label. In otherwords, semantic segmentation is a pixel-wise classification problem (solving many tiny classification problems).

## How do we evaluate this task?

There are five most common metrics: 1. Intersection-over-Union (IoU), 2. Mean Accuracy (mAcc), 3. Mean Intersection-over-Union (mIoU), 4. All Accuracy/Pixel Accuracy (allAcc),  5. frequency weighted IoU (fwIoU). More formally, consider two vectors, one containing predictions, and the other containing ground truth labels. Let $$n_{ij}$$ be the number of pixels of class $$i$$ predicted to
belong to class $$j$$, where there are $$n_{cl}$$ different classes, and let $$t_i = \sum\limits_j n_{ij}$$ be the total number of pixels of class $$i$$. We compute [3]:

$$
\begin{aligned}
mAcc &= \frac{1}{n_{cl}} \sum\limits_i \frac{ n_{ii} }{t_i} \\
mIoU &= \frac{1}{n_{cl}} \sum\limits_i \frac{ n_{ii} }{t_i + \sum\limits_j n_{ji} - n_{ii} } \\
allAcc &= \sum\limits_i \frac{ n_{ii} }{ \sum\limits_i t_i} \\
fwIoU &= \big( \sum\limits_k t_k \big)^{-1} \sum\limits_i \frac{ t_i n_{ii} }{t_i + \sum\limits_j n_{ji} - n_{ii} } \\
\end{aligned}
$$


We'll need basic tools from set theory to reason about discrete sets.

### Intersection-over-Union (IoU), also known as *Jaccard Index*

Consider two finite sample sets, $$A,B$$. The IoU is defined as the size of the intersection divided by the size of the union of the sample sets:

$$
IoU(A,B) = J(A,B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}
$$


<div class="fig figcenter fighighlight">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/Intersection_of_sets_A_and_B.svg/400px-Intersection_of_sets_A_and_B.svg.png" width="25%">
  <div class="figcaption">
   A graphical visualization of the intersection (light purple) and union (yellow, light purple, and orange combined) of two sets A,B. Source: [1].
  </div>
</div>

However, in segmentation, our predicted categories cannot be thrown into a big bag (set) and then compared with another big bag (set) of ground truth categories. Rather, these values must be compared at the exact corresponding spatial location, meaning we end up reasoning over many small sets (at each grid cell).


### Mean Intersection-over-Union (mIoU)

In a multi-class segmentation scenario, we wish to reason about the quality of our segmentation for each invididual class, and then coalesce these into a single representative number to summarize all of our information. We'll do this by computing intersection-over-union (IoU) for each class, and then averaging all IoUs into a mean IoU (mIoU).

Consider a 3-class problem, over $$2x2$$ grayscale images.

$$
\begin{equation}
\begin{array}{ll}
y_{pred} = \begin{bmatrix} 
2 & 0 \\ 1 & 0 
\end{bmatrix}, & & y_{true} = \begin{bmatrix} 
2 & 0 \\ 1 & 1 
\end{bmatrix},
\end{array}
\end{equation}
$$

We will be reasoning about each matrix cell individually, so we can reshape (flatten) these arrays and the evaluation result won't change.

$$
\begin{equation}
\begin{array}{ll}
y_{pred} = \begin{bmatrix} 
2 \\ 0 \\ 1 \\ 0 
\end{bmatrix}, & & y_{true} = \begin{bmatrix} 
2 \\ 0 \\ 1 \\ 1 
\end{bmatrix},
\end{array}
\end{equation}
$$

Let's compare horizontally-corresponding cells. We can think of each cell in the two column vectors as set, denoted with $$\{ \cdot \}$$ notation:

$$
\definecolor{red}{RGB}{255,0,0}
\definecolor{green}{RGB}{0,150,0}
\begin{equation}
\begin{array}{ll}
y_{pred} = \begin{bmatrix} 
\{2\} \\ \{0\} \\ \{1\} \\ \{0 \}
\end{bmatrix}, & & y_{true} = \begin{bmatrix} 
\{2\} \\ \{0\} \\ \{1\} \\ \{1\} 
\end{bmatrix},  & & (y_{true} + y_{pred}) = \begin{bmatrix} 
\color{green} \{2,2\}  \\ \color{green} \{0,0\}  \\ \color{green} \{1,1\}  \\ \color{green} \{0,1\}  \\
\end{bmatrix}, & & (y_{true} \cap y_{pred}) =
\begin{bmatrix}
\{2\} \cap \{2\} \\
\{0\} \cap \{0\} \\
\{1\} \cap \{1\} \\
\{0\} \cap \{1\} \\
\end{bmatrix} = 
 \begin{bmatrix} 
\color{red}  \{2\}  \\ \color{red} \{0\}  \\  \color{red} \{1\}  \\ \color{red} \emptyset \\
\end{bmatrix},
\end{array}
\end{equation}
$$

To find the union of these two sets, we'll add $$y_{pred}$$ and $$y_{true}$$ and then subtract the intersection:

$$
\definecolor{red}{RGB}{255,0,0}
\definecolor{green}{RGB}{0,150,0}
\begin{equation}
\begin{aligned}
\begin{array}{ll}
\big(y_{true} + y_{pred}\big) &- \big(y_{true} \cap y_{pred}\big) &= \big(y_{true} \cup y_{pred}\big)\\
 \begin{bmatrix} 
\color{green} \{2,2\}  \\ \color{green} \{0,0\}  \\ \color{green} \{1,1\}  \\ \color{green} \{0,1\}  \\
\end{bmatrix} &- \begin{bmatrix} 
\color{red} \{2\}  \\ \color{red} \{0\}  \\  \color{red} \{1\}  \\ \color{red} \emptyset \\
\end{bmatrix} &= \begin{bmatrix} 
\{2\} \\ \{0\} \\ \{1\} \\ \{0,1 \}
\end{bmatrix}
\end{array}
\end{aligned}
\end{equation}
$$

Now that we have computed intersection and union values enforcing the spatially corresponding locations, we can throw all values into a big bag and assess.


**Implementation In Numpy [2]:**

We'll start with two tensors: a model `output` vector (predictions) and a `target` vector. They must be of the same size:
```python
assert output.shape == target.shape
```
We can arbitrarily flatten them both to 1d arrays, since this will preserve the cell correspondences.
```python
output = output.reshape(output.size).copy()
target = target.reshape(target.size)
```
We end up with two column vectors:

 `output = array([2, 0, 1, 0])` and `target = array([2, 0, 1, 1])`.

We seek to know the values in cells where `output` and `target` are identical (intersection values):
```python
intersection = output[np.where(output == target)[0]]
```
Thus `intersection = array([2, 0, 1])`.

We'll use `np.histogram` to count the number of samples in each bin for each vector. `np.histogram` accepts as an input of the histogram bin edges. The bin edges must be in monotonically increasing order, and include the rightmost edge.

In a 3-class example, our three bins will be the ranges [0,1], [1,2], and [2,3], so our bin edges will be array([0, 1, 2, 3]):

```python
area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
```
We find `intersection` values fall into 3 bins: `area_intersection = array([1, 1, 1])`

```python
area_output, _ = np.histogram(output, bins=np.arange(K+1))
area_target, _ = np.histogram(target, bins=np.arange(K+1))
```
Our bin counts for the two vectors are:

`area_target = array([1, 2, 1])` and `area_output = array([2, 1, 1])`

```python
area_union = area_output + area_target - area_intersection
```

Sure enough, our bin counts for `area_union` are `array([2, 2, 1])` for respective classes 0,1,2, just as we expected if we were to bin the following vector:

$$
\begin{bmatrix} 
\{2\} \\ \{0\} \\ \{1\} \\ \{0,1 \}
\end{bmatrix}
$$

Now that we have cardinality counts for intersection and union for each of the 3 classes, we can find the mIoU by computing `miou = np.mean(area_intersection / (area_union + 1e-10))`
where $$1^{-10}$$ is used to prevent division by zero. Our mIoU would be 0.66, since our per-class IoUs are `array([0.5, 0.5, 1. ])`.

Putting the code all together:

```python
def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    # flatten the tensors to 1d arrays
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = 255
    intersection = output[np.where(output == target)[0]]
    # contain the number of samples in each bin.
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target
```




## Mean Accuracy (mAcc)


## All Accuracy (allAcc)

## Frequency Weighted Intersection-over-Union (fwIoU)



We often have a background label we'll ignore, so identify the locations in the label ("target") where the ground truth class was void/background/unlabeled (=255, in our case) and set each value of our predictions (model "output") to 255 also, to ignore these values.  




area_intersection will hold counts of correct predictions for each class.
area_output will hold counts of predictions for each class.
area_target will hold counts of how many pixels belong to each class in the ground truth.




```
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
```


```python
def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = 255
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target
```

## State-of-the-Art Method: Pyramid Scene Parsing Network (PSPNet)



```python
import torch
from torch import nn
import torch.nn.functional as F
import model.resnet as models
```


## The Pyramid Pooling Module

Effective global contextual prior... 

a hierarchical global prior, containing information with different scales and varying among different sub-regions.

The pyramid pooling module fuses features under four different pyramid scales.

The coarsest level highlighted in red is global pooling to generate a single bin output. The following pyramid level separates the feature map into different sub-regions and forms pooled representation for different locations. The output of different levels in the pyramid pooling module contains the feature map with varied sizes. To maintain the weight of global feature, we use 1×1 convolution layer after each pyramid level to reduce the dimension of context representation to 1/N of the original one if the level size of pyramid is N

e directly upsample the low-dimension feature maps to get the same size feature as the original feature map via bilinear interpolation

Our pyramid pooling module (PPM) is a four-level one with bin sizes of 1×1, 2×2, 3×3 and 6×6 respectively. 


In short, the PPM utilizes `nn.AdaptiveAvgPool2d(bin)` to break an image into (bin x bin) subregions, and then pools all entries inside each subregion.

For a $$1 \times 1$$ bin, we simply obtain the average pixel value for a 1-channel feature map:
```python
>>> import torch
>>> pool = torch.nn.AdaptiveAvgPool2d(1)
>>> x = torch.tensor([[1.,2.],[3.,4.]])
>>> x = x.reshape(1,1,2,2)
>>> x
tensor([[[[1., 2.],
          [3., 4.]]]])
>>> x.shape
torch.Size([1, 1, 2, 2])
>>> pool(x)
tensor([[[[2.5000]]]])
>>> (1 + 2  + 3 + 4) / 4
2.5
```

For an output size of $$3 \times 2$$, we'll find the following behavior:

```python
>>> import numpy as np
>>> import torch
>>> x = torch.from_numpy(np.array(range(48)))
>>> x = x.type(torch.FloatTensor)
>>> x
tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
        14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27.,
        28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 40., 41.,
        42., 43., 44., 45., 46., 47.])
>>> x = x.reshape(6,8)
>>> pool = torch.nn.AdaptiveAvgPool2d((3,2))

>>> x.shape
torch.Size([6, 8])
>>> x = x.reshape(1,1,6,8)
>>> x.shape
torch.Size([1, 1, 6, 8])
>>> pool(x)
tensor([[[[ 5.5000,  9.5000],
          [21.5000, 25.5000],
          [37.5000, 41.5000]]]])
>>> x
tensor([[[[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.],
          [ 8.,  9., 10., 11., 12., 13., 14., 15.],
          [16., 17., 18., 19., 20., 21., 22., 23.],
          [24., 25., 26., 27., 28., 29., 30., 31.],
          [32., 33., 34., 35., 36., 37., 38., 39.],
          [40., 41., 42., 43., 44., 45., 46., 47.]]]])
```

We can easily verify the first three entries of the pooled feature map:

```python
>>> ( 0 + 1 + 2 + 3 + 8 + 9 + 10 + 11 ) / 8
5.5
>>> (16 + 17 + 18 + 19 + 24 + 25 + 26 + 27) / 8
21.5
>>> (32 + 33 + 34 + 35 + 40 + 41 + 42 + 43)/8
37.5
```


```python
fea_dim = 2048
bins=(1, 2, 3, 6)
self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins, BatchNorm)
```

```python
class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins, BatchNorm):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                BatchNorm(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)
```


```python
class PSPNet(nn.Module):
    def __init__(self, layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=8, use_ppm=True, criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, pretrained=True):
        super(PSPNet, self).__init__()
        assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion
        models.BatchNorm = BatchNorm

        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        fea_dim = 2048
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins, BatchNorm)
            fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            BatchNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                BatchNorm(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )
```





```python
    def forward(self, x, y=None):
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)
        if self.use_ppm:
            x = self.ppm(x)
        x = self.cls(x)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            aux = self.aux(x_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)
            return x.max(1)[1], main_loss, aux_loss
        else:
            return x
```
## DeepLabv3

The Atrous Spatial Pyramid Pooling (ASPP) module, as implemented [here](https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/deeplabv3.py#L64).

```python
class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
```


## Separable vs. A Trous Convolutions

DeepLab dataset and paper




L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille. Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully con- nected crfs. IEEE transactions on pattern analysis and ma- chine intelligence, 40(4):834–848, 2018.

## References

[1].  Wikipedia. *Jaccard Index*. [Link](https://upload.wikimedia.org/wikipedia/commons/thumb/e/ee/Union_of_sets_A_and_B.svg/400px-Union_of_sets_A_and_B.svg.png).

[2]. Hengshuang Zhao. `semseg` repository. [Link](https://github.com/hszhao/semseg/blob/master/util/util.py).

[3] Jonathan Long, Evan Shelhamer, Trevor Darrell. *Fully Convolutional Networks for Semantic Segmentation*. CVPR 2015. [PDF](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf).

[4] Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam. Rethinking Atrous Convolution for Semantic Image Segmentation. [PDF](https://arxiv.org/abs/1706.05587).
