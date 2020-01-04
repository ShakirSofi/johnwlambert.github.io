---
layout: post
title:  "Advanced Stereo Topics"
permalink: /advanced-stereo/
excerpt: "Advanced Stereo Topics"
mathjax: true
date:   2018-12-27 11:00:00
mathjax: true

---
Table of Contents:


## 



Dynamic programming, used in this case. Use a single match as constraint to find other matches. get ordering constraint.

sometimes blue patch is not going to happen! happens all the time. have to allow that sometime there will not be a match, and just move on to the next guy.
 
real-time stereo in late 80s from stereo matching with dynamic programming.

Turns out there are quite a few problems.

illusion? thin nail illusion, thin object close to the camera?

slanted plane: scrunched, will have multiple matches in one location.

one scanline at a time, wasnt the winning approach.

matching the scannline independently isn't good -- may see weird artifacts when you look at a vertical column, 3d boundaries in scene may not make straight lines. no consistently through your scanlines.

Scanline is bad as representation of the world. segemntation based stereo -- first segment the iamge into patches that are similar in their appearance, then work patch by patch and do the matching.

you can also get view interpolation, synthesize new views in between the two views.




## Simple Example on KITTI

$$
E = R \mbox{ } [t]_{\times}
$$



## Object Detection in Stereo

Chen *et al.*
3DOP (NIPS 2015) Urtasun [4]

## Unsupervised Learning of Depth

From 2017 [5]

Let $$p_t$$ denote homogeneous coordinates in the target view. $$K$$ denotes the camera intrinsic matrix.

$$
p_s \sim K \hat{T}_{t \rightarrow s} \hat{D}_t (p_t) K^{-1} p_t
$$


[4] X. Chen, K. Kundu, Y. Zhu, A. Berneshawi, H. Ma, S. Fidler, R. Urtasun. *3D Object Proposals for Accurate Object Class Detection.*  Advances in Neural Information Processing Systems 28 (NIPS 2015). [PDF](https://papers.nips.cc/paper/5644-3d-object-proposals-for-accurate-object-class-detection).
