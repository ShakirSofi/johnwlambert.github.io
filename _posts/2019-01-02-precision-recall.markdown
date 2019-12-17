---
layout: post
title:  "Recall and Precision"
permalink: /recall-precision/
excerpt: " "
mathjax: true
date:   2018-12-27 11:00:00
mathjax: true

---
Table of Contents:
- [Recall](#recall)
- [Precision](#precision)
- [Mean Average Precision (mAP)](#map)


<a name='need-for-dr'></a>

Average Precision is very common in two fields (1) object detection, (2) keypoint/patch detection + description, and (3) information retrieval. We will be focused only on binary relevance.

## The Need for More Finely-Grained Measures of Accuracy

We'll suppose that we are performing binary classification: classifying objects into different classes.

Mean Average Precision involves computing the area under a curve (an integral), and can actually be quite confusing.

### Recall

Recall measures how many objects you missed. You can have very high recall by classifying everything as some class (at the expense of precision). It is a proportion to measure, *How much of the good stuff did we miss?*

$$
\mbox{Recall} = \frac{tp}{tp+fn} = \frac{tp}{\mbox{no. of actual positives}}
$$

### Precision

Precision measures your discriminative ability. If you claimed that all of the objects you saw were of a particular class, and you were usually wrong because they belonged to a different class, you would have low precision. Your judgments can't be considered *precise*. Precision is a proportion to measure, *How much junk did we give to the user?*

$$
\mbox{Precision} = \frac{tp}{tp+fp} = \frac{tp}{\mbox{no. of predicted positives}}
$$



## Trading Off Between the Two
Since these two metrics are complementary, we must always report them together. If we return only one of these two numbers, we can make either number arbitrarily high.

In an object detection system, we may require different amounts of overlap with the ground truth box to assert that a detection is a true positive detection. That overlap threshold may be ours to choose. As we increase the threshold, say from 0.7 to 0.9, we get higher precision, but worse recall (only making prediction if we are pretty confident). However, suppose we wish to avoid false negatives at all costs -- if we are blind to a single object detection, we may collide with it and damage our robot/vehicle. In this scenario, we would probably opt for a lower threshold, perhaps 0.3. Our classifier will now have higher recall, but lower precision. 

As we vary the value of this threshold, we can actually draw a curve -- a precision-recall curve.

## Accuracy Isn't Helpful
Classes are very imbalanced. You may get 99.99% accuracy for a search algorithm by predicting (or retrieving) nothing. In object detection, there are an infinite number of bounding boxes you could predict. If there is only one object, predicting nothing, you would have great accuracy. In IR, for any query, almost all documents in a corpus are not relevant. The true negatives (which can measure in the billions) are the things we don't care about, and which we can ignore in precision/recall.


## Information Retrieval (IR)
In IR, given a user query, an IR system will retrieve documents from a corpus (predictions). We will compare this set with the documents relevant to the user (positives). Thus, a true positive is a *relevant document* with respect to a query (accurately retrieved document). False negatives are relevant documents that your system missed. And False positives are documents your system should not have retrieved.

## Recall and Precision over Ranks
In object detection, we generally have confidences associated with each prediction. In IR, we will be also be able to rank search results by confidence. While F-measure is one way of combining recall and precision, it cannot incorporate ranking or confidence information (at many different thresholds?) into its evaluation.

You can compute precision and recall at each ranking. You can compute precision and recall at each ranking. Generally we try to plot these numbers.

Precision @ K. We set a rank threshold @ K, and compute the % of relevant in the top K. We ignore all documents ranked lower than K. For example,



We can do the same for Recall @K.

Suppose for a query (seeking 3 ground truth documents), we return 5 documents. Let $$\color{limegreen}\blacksquare$$ represent a relevant document (TP), and let $$\color{red}\blacksquare$$ represent an irrelevant document (false positive). Suppose our 5 documents are ranked as follows:

$$
\color{limegreen}\blacksquare \hspace{1mm} \color{red}\blacksquare \hspace{1mm} \color{limegreen}\blacksquare \hspace{1mm} \color{red}\blacksquare \hspace{1mm} \color{limegreen}\blacksquare
$$

Our Prec@1 is 1/1, our Prec@2 is 1/2, Prec@3 is 2/3, our Prec@4 is 2/4, and our Prec@5 is 3/5. Our Recall@1 is 1/3, Recall@2 is 1/3, Recall@3 is 2/3, Recall@4 is 2/3, Recall@5 is 3/3.
```python
prec = [1/1, 1/2, 2/3, 2/4, 3/5]
recall = [1/3, 1/3, 2/3, 2/3, 3/3]
plt.plot(range(5), prec, c='r', label='Precision'); 
plt.scatter(range(5), prec, 50, marker='.', c='r')
plt.plot(range(5), recall, c='b', label='Recall'); 
plt.scatter(range(5), recall, 50, marker='.', c='b')
plt.xlabel('Rank'); plt.show()
```
Recall can never go down.


## Average Precision
If we consider the rank position of each relevant document, $$K_1, K_2, \dots, K_R$$, then we can compute the Precision@K for each $$K_1, K_2, \dots, K_R$$. The average precision is the average of all P@K values. 





## Mean Average Precision

When performing the task of object detection, we would like to be discriminative not just about two classes




```python
# compute precision recall
fp = np.cumsum(fp)
tp = np.cumsum(tp)
rec = tp / float(npos)
# avoid divide by zero in case the first detection matches a difficult
# ground truth
prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
ap = voc_ap(rec, prec, use_07_metric)
```


Here is PASCAL VOC's AP computation from [Detectron2](https://github.com/facebookresearch/detectron2/blob/master/detectron2/evaluation/pascal_voc_evaluation.py)
```python
def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
```






## Object Detection Example
In order to compute average precision, we first need to be able to count our true positives and false positives. To do so, we will compute overlap (i.e. IoU). Let ovmax be the maximum det-gt overlap observed thus far. In addition, if you have more than one detection for a single ground truth object, this must be considered an error (false positive). Here is the pseudocode:
```python
sort detections from greatest to smallest confidence
for each det in detections:
    ovmax =  -infinity
    for each gt in ground truth boxes:
        overlap = compute overlap between det and gt
        if overlap > ovmax:
            ovmax = overlap
            best_gt_idx = gt index

    if ovmax >= minimum_overlap_threshold:
        if gt bbox already claimed:
            mark det as false positive
        else:
            mark det as true positive
```
Here is the actual code, as found in [Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py) or in FAIR's [Detectron2](https://github.com/facebookresearch/detectron2/blob/master/detectron2/evaluation/pascal_voc_evaluation.py).

```python
def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print 'Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames))
        # save
        print 'Saving cached annotations to {:s}'.format(cachefile)
        with open(cachefile, 'w') as f:
            cPickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'r') as f:
            recs = cPickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap
```

## Now for COCO:

```python
# inspired from Detectron:
# https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L255 # noqa
def _evaluate_box_proposals(dataset_predictions, coco_api, thresholds=None, area="all", limit=None):
    """
    Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "96-128": 4,
        "128-256": 5,
        "256-512": 6,
        "512-inf": 7,
    }
    area_ranges = [
        [0 ** 2, 1e5 ** 2],  # all
        [0 ** 2, 32 ** 2],  # small
        [32 ** 2, 96 ** 2],  # medium
        [96 ** 2, 1e5 ** 2],  # large
        [96 ** 2, 128 ** 2],  # 96-128
        [128 ** 2, 256 ** 2],  # 128-256
        [256 ** 2, 512 ** 2],  # 256-512
        [512 ** 2, 1e5 ** 2],
    ]  # 512-inf
    assert area in areas, "Unknown area range: {}".format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = []
    num_pos = 0

    for prediction_dict in dataset_predictions:
        predictions = prediction_dict["proposals"]

        # sort predictions in descending order
        # TODO maybe remove this and make it explicit in the documentation
        inds = predictions.objectness_logits.sort(descending=True)[1]
        predictions = predictions[inds]

        ann_ids = coco_api.getAnnIds(imgIds=prediction_dict["image_id"])
        anno = coco_api.loadAnns(ann_ids)
        gt_boxes = [
            BoxMode.convert(obj["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            for obj in anno
            if obj["iscrowd"] == 0
        ]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
        gt_boxes = Boxes(gt_boxes)
        gt_areas = torch.as_tensor([obj["area"] for obj in anno if obj["iscrowd"] == 0])

        if len(gt_boxes) == 0 or len(predictions) == 0:
            continue

        valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
        gt_boxes = gt_boxes[valid_gt_inds]

        num_pos += len(gt_boxes)

        if len(gt_boxes) == 0:
            continue

        if limit is not None and len(predictions) > limit:
            predictions = predictions[:limit]

        overlaps = pairwise_iou(predictions.proposal_boxes, gt_boxes)

        _gt_overlaps = torch.zeros(len(gt_boxes))
        for j in range(min(len(predictions), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)
    gt_overlaps = torch.cat(gt_overlaps, dim=0)
    gt_overlaps, _ = torch.sort(gt_overlaps)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {
        "ar": ar,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
    }
```


## References

[1] Bharath Hariharan and Ross Girshick. Fast/er R-CNN. [https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py](https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py).

https://en.wikipedia.org/w/index.php?title=Information_retrieval&oldid=793358396#Average_precision


https://www.youtube.com/watch?v=yjCMEjoc_ZI

[2] Victor Lavrenk Lecture, Univ. Edinburgh. []

[3] Victor Lavrenko Lecture, Univ. Edinburgh. [Why We Can't Use Accuracy](https://www.youtube.com/watch?v=mYW0PDnuPm0&list=PLBv09BD7ez_6nqE9YU9bQXpjJ5jJ1Kgr9&index=7)

[3] Victor Lavrenko Lecture, Univ. Edinburgh. [Evaluation 10: recall and precision over ranks.](https://www.youtube.com/watch?v=H7oAofuZjjE&list=PLBv09BD7ez_6nqE9YU9bQXpjJ5jJ1Kgr9&index=10)

[4]

[5]




