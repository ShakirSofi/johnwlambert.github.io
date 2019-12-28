---
layout: post
title:  "Geometric Transformations"
permalink: /geometric-transform/
excerpt: "sheer, spatial transformers, homographies"
mathjax: true
date:   2019-12-24 11:00:00

---
Table of Contents:
- [Why do we need residual connections?](#need-for-residual)



## Transformation Classes

sheer, affine, etc.

## Spatial Transformer


## Homography

The motion of a plane can be described by a homography. This is true for a moving plane imaged by a static camera. It is true for a static plane imaged by a moving camera [2].

video stabilization
F-matrix is special case

## 3x3 Homography Parameterization
The simplest way to parameterize a homography is with a 3x3 matrix and a fixed scale. The homography maps $$[u, v]$$, the pixels in the left image, to $$[u^{\prime}, v^{\prime}]$$, the pixels in the right image, and is defined up to scale:

$$
\begin{bmatrix} u^{\prime} \\ v^{\prime} \\1 \end{bmatrix}
= \begin{bmatrix}
H_{11} & H_{12} & H_{13} \\
H_{21} & H_{22} & H_{23} \\
H_{31} & H_{32} & H_{33} \\
\end{bmatrix}
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
$$

the submatrix $$ \begin{bmatrix} H_{11} & H_{12} \\ H_{21} & H_{22} \end{bmatrix}$$, represents the rotational terms in the homography, while the vector $$\begin{bmatrix}H_{13} \\ H_{23} \end{bmatrix}$$ is the translational offset.

## 4-Point Homography Parameterization

Baker et al. [2] dicuss the 4pt parameterization.

## examples

[1 0 0]
[0 1 0]
[0 0 1]

[s 0 0
[0 s 0]
[0 0 1]

[1 0 tx]
[0 1 ty]
[0 0 1]


[R   0]
[      0
[0 0 1]

## References

[1] Detone, Malisiewicz, Rabinovich. Deep Image Homography Estimation.

[2] Simon Baker, Ankur Datta, and Takeo Kanade. Parameterizing homographies. Technical Report CMU-RI-TR06-11, Robotics Institute, Pittsburgh, PA, March 2006.


