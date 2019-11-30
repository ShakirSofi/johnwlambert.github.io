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
  <img src="/assets/stereo_coordinate_systems_rehg_dellaert.jpg" width="90%">
  <div class="figcaption">
    Two cameras with optical centers O_L and O_R are separated by a baseline B. The z-axis extends towards the world, away from the camera.
  </div>
</div>

Now, consider just the plane spanned by the $$x$$- and $$z$$-axes, with a constant $$y$$ value:
<div class="fig figcenter fighighlight">
  <img src="/assets/stereo_vision_setup.jpg" width="50%">

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



## SSD or SAD is only the beginning

SSD/SAD suffers from... large "blobs" of disparities, can have areas of large error in textureless regions. fail in shadows

MC-CNN can... show fine structures. succeed in shadows.

Too small a window might not be able to distinguish unique features of an image, but too large a window would mean many patches would likely have many more things in common, leading to less helpful matches.



## Cost Volumes

Appendix B.5 and a recent survey paper on MRF inference (Szeliski, Zabih, Scharstein et al. 2008)


## Simple Example on KITTI

$$
E = R \mbox{ } [t]_{\times}
$$



## When is smoothing a good idea?

Of course smoothing should help in noisy areas of an image. It will only help to alleviate particular types of noise, however. It is not a definitive solution to obtaining better depth maps.

Smoothing performs best on background/uniform areas: disparity values here are already similar to each other, so smoothing doesn't have to make any drastic changes. Pixels near each other are actually supposed to have similar disparity values, so the technique makes sense in these regions. Smoothing can also help in areas of occlusion, where SAD may not be able to find any suitable match. Smoothing will curb the problems by pushing the dispairty of a pixel to be similar to that of its neighbors.

Smoothing performs poorly in areas of fine detail, with narrow objects. It also performs poorly in areas with many edges and depth discontinuities. Smoothing algorithms penalize large disparity differences in closeby regions (which can actually occur in practice). Smoothing penalizes sharp disparity changes (corresponding to depth discontinuities).


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


Computing the Stereo Matching Cost with a Convolutional Neural Network. Jure Zbontar  and Yann LeCun. [PDF](https://arxiv.org/pdf/1409.4326.pdf).

[1] Richard Szeliski. 

[2] James Hays. [PDF](https://www.cc.gatech.edu/~hays/compvision/lectures/09.pdf).

[3] Rajesh Rao. Lecture 16: Stereo and 3D Vision, University of Washington. [PDF](https://courses.cs.washington.edu/courses/cse455/09wi/Lects/lect16.pdf).

[4] X. Chen, K. Kundu, Y. Zhu, A. Berneshawi, H. Ma, S. Fidler, R. Urtasun. *3D Object Proposals for Accurate Object Class Detection.*  Advances in Neural Information Processing Systems 28 (NIPS 2015). [PDF](https://papers.nips.cc/paper/5644-3d-object-proposals-for-accurate-object-class-detection).

[5]

[PDF](http://openaccess.thecvf.com/content_cvpr_2017/html/Zhou_Unsupervised_Learning_of_CVPR_2017_paper.html).


STEREO http://people.scs.carleton.ca/~c_shu/Courses/comp4900d/notes/simple_stereo.pdf
DEPTH http://www.cse.usf.edu/~r1k/MachineVisionBook/MachineVision.files/MachineVision_Chapter11.pdf
STEREO https://courses.cs.washington.edu/courses/cse455/09wi/Lects/lect16.pdf
SFM http://cvgl.stanford.edu/teaching/cs231a_winter1415/lecture/lecture6_affine_SFM_notes.pdf
JAMES MULTI-VIEW https://www.cc.gatech.edu/~hays/compvision/lectures/09.pdf



