---
layout: post
title:  "Stereo and Disparity"
permalink: /stereo/
excerpt: "focal length, similar triangles, depth "
mathjax: true
date:   2018-12-27 11:00:00
mathjax: true

---
Table of Contents:
- [Stereo Matching](#sfmpipeline)
- [Disparity](#costfunctions)
- [Bundle Adjustment](#bundleadjustment)

<a name='sfmpipeline'></a>



## Stereo Vision and Stereo Matching

"Stereo matching" is the task of estimating a 3D model of a scene from two or more images. The task requires finding matching pixels in the two images and converting the 2D positions of these matches into 3D depths [1]

Humans have stereo vision with a baseline of 60 mm.



## Disparity

Consider a simple model of stereo vision: we have two cameras whose optic axes are parallel. Each camera points down the $$Z$$-axis. A figure is shown below:

<div class="fig figcenter fighighlight">
<<<<<<< HEAD
  <img src="/assets/stereo_vision_setup.jpg" width="50%">
=======
  <img src="/assets/stereo_vision_setup.png" width="50%">
>>>>>>> bbabd62b79acfce4d33cc330fc01a5e0b1e16d7f
  <div class="figcaption">
    Two cameras L and R are separated by a baseline b. Here the Y-axis is perpendicular to the page. f is our (horizontal) focal length.
  </div>
</div>
In this figure, the world point $$P=(x,z)$$ is projected into the left image as $$p_l$$ and into the right image as $$p_r$$.

By defining right triangles we can find two similar triangles: with vertices at $$(0,0)-(0,z)-(x,z)$$ and $$(0,0)-(0,f)-(f,x_l)$$. Since they share the same angle $$\theta$$, then $$\mbox{tan}(\theta)= \frac{\mbox{opposite}}{\mbox{adjacent}}$$ for both, meaning:

$$
\frac{z}{f} = \frac{x}{x_l}
$$

We notice another pair of similar triangles
$$(b,0)-(b,z)-(x,z)$$ and $$(b,0)-(b,f)-(b+x_r,f)$$, which by the same logic gives us

$$
\frac{z}{f} = \frac{x-b}{x_r}
$$

We'll derive a closed form expression for depth in terms of disparity. We already know that

$$\frac{z}{f} = \frac{x}{x_l}$$

Multiply both sides by $$f$$, and we get an expression for our depth from the observer:

$$z = f(\frac{x}{x_l})$$

We now want to find an expression for $$\frac{x}{x_l}$$ in terms of $$x_l-x_r$$, the *disparity* between the two images. 

$$
\begin{align}
\frac{x}{x_l} &= \frac{x-b}{x_r} & \text{By similar triangles} \\
x x_r &= x_l (x-b) & \text{Multiply diagonals of the fraction} \\
x x_r &= x_lx - x_l b & \text{Distribute terms} \\
x_r &= \frac{x_lx}{x} - b(\frac{x_l}{x}) & \text{Divide all terms by } x \\
x_r &= x_l - b(\frac{x_l}{x}) & \text{Simplify} \\
b\frac{x_l}{x} &= x_l - x_r & \text{Rearrange terms to opposite sides} \\
b (\frac{x_l}{x}) (\frac{x}{x_l}) &= (x_l - x_r) (\frac{x}{x_l}) & \text{Multiply both sides by fraction inverse} \\
b &= (x_l - x_r) (\frac{x}{x_l}) & \text{Simplify} \\
\frac{b}{x_l - x_r} &= \frac{x}{x_l} & \text{Divide both sides by } (x_l - x_r) \\
\end{align}
$$

We can now plug this back in

$$z = f(\frac{x}{x_l}) = f(\frac{b}{x_l - x_r}) $$

What is our takeaway? The amount of horizontal distance between the object in Image L and image R (*the disparity* $$d$$) is inversely proportional to the distance $$z$$ from the observer. This makes perfect sense. Far away objects (large distance from the observer) will move very little between the left and right image. Very closeby objects (small distance from the observer) will move quite a bit more. The focal length $$f$$ and the baseline $$b$$ between the cameras are just constant scaling factors.

We made two large assumptions:

1. We know the focal length $$f$$ and the baseline $$b$$. This requires prior knowledge or camera calibration.
2. We need to find point correspondences, e.g. find the corresponding $$(x_r,y_r)$$ for
each $$(x_l,y_l)$$.

## The Epipolar Line, Plane, and Constraint

Unfortunately, just because we know

  how to compute for a given pixel in one image the range of possible locations the pixel might appear at in the other image, i.e., its epipolar lin



## Sum of Squared-Differences


The matching cost is the squared difference of intensity values at a given disparity.


## Cost Volumes

Appendix B.5 and a recent survey paper on MRF inference (Szeliski, Zabih, Scharstein et al. 2008)


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


## References

[1] Richard Szeliski. 

[2] James Hays. [PDF](https://www.cc.gatech.edu/~hays/compvision/lectures/09.pdf).

[3] Rajesh Rao. Lecture 16: Stereo and 3D Vision, University of Washington. [PDF](https://courses.cs.washington.edu/courses/cse455/09wi/Lects/lect16.pdf).

[4] X. Chen, K. Kundu, Y. Zhu, A. Berneshawi, H. Ma, S. Fidler, R. Urtasun. *3D Object Proposals for Accurate Object Class Detection.*  Advances in Neural Information Processing Systems 28 (NIPS 2015). [PDF](https://papers.nips.cc/paper/5644-3d-object-proposals-for-accurate-object-class-detection).

[5]

[PDF](http://openaccess.thecvf.com/content_cvpr_2017/html/Zhou_Unsupervised_Learning_of_CVPR_2017_paper.html).


