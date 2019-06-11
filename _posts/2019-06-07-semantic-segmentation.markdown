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

The goal is to assign each pixel in the image a category label.

## How do we evaluate this task?


1. Intersection-over-Union (IoU), also known as *Jaccard Index*


<div class="fig figcenter fighighlight">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/Intersection_of_sets_A_and_B.svg/400px-Intersection_of_sets_A_and_B.svg.png" width="65%">
  <div class="figcaption">
   CAPTION GOES HERE
  </div>
</div>


https://upload.wikimedia.org/wikipedia/commons/thumb/e/ee/Union_of_sets_A_and_B.svg/400px-Union_of_sets_A_and_B.svg.png

2. Mean Intersection-over-Union (mIoU)

3. Mean Accuracy (mAcc)

4. allAcc

We often have a background label we'll ignore, so identify the locations in the label ("target") where the ground truth class was void/background/unlabeled (=255, in our case) and set each value of our predictions (model "output") to 255 also, to ignore these values.  

?? WONT THIS INFLATE THINGS?

Flatten all arrays to 1d. We then find where the output equals the ground truth target:
```python
np.where(output == target)[0]
```

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
