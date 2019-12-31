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
