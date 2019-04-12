---
layout: post
title:  "Pose Graph SLAM"
permalink: /pose-slam/
excerpt: "Simultaneous Localization and Mapping, GraphSLAM, loop closures"
mathjax: true
date:   2018-12-27 11:00:00
mathjax: true

---
Table of Contents:
- [Gauss-Newton Optimization](#sfmpipeline)
- [2D SLAM: Pose-Pose Constraints](#2d-pose-pose)
- [2D SLAM: Pose-Landmark Constraints ](#2d-pose-landmark)

<a name='sfmpipeline'></a>

## Pose Graph SLAM



## Review of Gauss-Newton Optimization

In vector calculus, the Jacobian matrix is the matrix of all first-order partial derivatives of a vector-valued function. In our case, the vector-valued function will be an error function. The entries of the Jacobian are defined as:

$$
J = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots 	& & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}
$$

## 2D-SLAM

### Pose-Pose Constraints

Consider the pose of a robot at time $$i$$: $$\mathbf{x}_i = \begin{bmatrix} x_i \\ y_i \\ \theta_i \end{bmatrix}$$. If we wish to convert from this pose vector to a homogeneous transform, we can use the function $$v2t(\cdot)$$, defined as:

$$
v2t(\mathbf{x}_i) = \begin{bmatrix} R_i & \mathbf{t}_i \\ 0 & 1 \end{bmatrix} = \begin{bmatrix}
\mbox{cos}(\theta_i) & -\mbox{sin}(\theta_i) & x_i \\ 
\mbox{sin}(\theta_i) & \mbox{cos}(\theta_i) & y_i \\ 
0 & 0 & 1
\end{bmatrix} = {}^wX_i
$$

Consider the error function (boldface indicates a vector)

$$
\mathbf{e}_{ij} = t2v\bigg(Z_{ij}^{−1} ({}^wX_i^{−1} {}^wX_j )\bigg) = \begin{bmatrix}
R_{ij}^T\Big( R_i^T(\mathbf{t}_j - \mathbf{t}_i)-\mathbf{t}_{ij}\Big) \\
\theta_j - \theta_i - \theta_{ij}
\end{bmatrix} = \begin{bmatrix} \mathbf{e}_{xy} \\ \mathbf{e}_{\theta} \end{bmatrix}
$$, 

where $$Z_{ij} = v2t(\mathbf{z}_{ij})$$ is the
transformation matrix of the measurement $$\mathbf{z}_{ij}^T = (\mathbf{t}_{ij}^T, \theta_{ij})$$. We can use the measured $$\theta_{ij}$$ to form another rotation matrix $$R_{ij}$$. In our case the Jacobian of the error function $$\mathbf{e}_{ij}$$ is a $$3 \times 3$$ matrix, but will be simpler to write in 4 blocks, instead of 9 entries:

$$
A_{ij} = \frac{\partial \mathbf{e}_{ij}}{\partial \mathbf{x}_i} = \begin{bmatrix}
\frac{\partial \mathbf{e}_x}{\partial \mathbf{x}_{i_x}} & \frac{\partial \mathbf{e}_x}{\partial \mathbf{x}_{i_y}} & \frac{\partial \mathbf{e}_x}{\partial \mathbf{x}_{i_{\theta}}} \\
\frac{\partial \mathbf{e}_y}{\partial \mathbf{x}_{i_x}} & \frac{\partial \mathbf{e}_y}{\partial \mathbf{x}_{i_y}} & \frac{\partial \mathbf{e}_y}{\partial \mathbf{x}_{i_{\theta}}} \\
\frac{\partial \mathbf{e}_{\theta}}{\partial \mathbf{x}_{i_x}} & \frac{\partial \mathbf{e}_{\theta}}{\partial \mathbf{x}_{i_y}} & \frac{\partial \mathbf{e}_{\theta}}{\partial \mathbf{x}_{i_{\theta}}}
\end{bmatrix} = \begin{bmatrix}
\frac{\partial \mathbf{e}_{xy}}{\partial \mathbf{t}_i} &  \frac{\partial \mathbf{e}_{xy}}{\partial \mathbf{x}_{i_{\theta}}} \\
\frac{\partial \mathbf{e}_{\theta}}{\partial \mathbf{t}_i} & \frac{\partial \mathbf{e}_{\theta}}{\partial \mathbf{x}_{i_{\theta}}}
\end{bmatrix}
$$

We will derive each of these 4 Jacobian terms by hand using matrix calculus.

1. First, note that $$\mathbf{e}_{xy} = R_{ij}^T\Big( R_i^T(\mathbf{t}_j - \mathbf{t}_i)-\mathbf{t}_{ij}\Big)$$, so if we look at the terms that are multiplied with $$\mathbf{t}_i$$, we find
$$\frac{\partial \mathbf{e}_{xy}}{\partial \mathbf{t}_i} = R_{ij}^T R_i^T(-1)$$.

2. Second, consider
$$
\frac{\partial \mathbf{e}_{xy}}{\partial \mathbf{x}_{i_{\theta}}}
$$. Since we find $$R_i^T$$ in $$\mathbf{e}_{xy}$$, and $$R_i$$ is a function of $$\theta$$, i.e. $$R(\theta_i)$$, there is a dependence. Note that we can find the derivative of our 2D rotation matrix $$R_i^T = \begin{bmatrix} \mbox{cos}(\theta_i) & \mbox{sin}(\theta_i) \\ -\mbox{sin}(\theta_i) & \mbox{cos}(\theta_i) \end{bmatrix}$$. Formally, since $$R_i^T$$ is a function of $$\theta_i$$, the derivative $$\frac{\partial R_{i}^T}{\partial \theta_i}$$ is defined as:
$$
\frac{\partial R_{i}^T}{\partial \theta_i} = \begin{bmatrix} \frac{\partial \text{cos}(\theta_i)}{\partial \theta_i } & \frac{\partial \text{sin}(\theta_i)}{\partial \theta_i } \\ -\frac{\partial \text{sin}(\theta_i)}{\partial \theta_i } & \frac{\partial \text{cos}(\theta_i)}{\partial \theta_i } \end{bmatrix} = \begin{bmatrix} -\mbox{sin}(\theta_i) & \mbox{cos}(\theta_i) \\ -\mbox{cos}(\theta_i) & -\mbox{sin}(\theta_i) & \end{bmatrix}
$$

3. Third, consider
$$
\frac{\partial \mathbf{e}_{\theta}}{\partial \mathbf{t}_i} = 
$$
Since $$ \mathbf{e}_{\theta} = \theta_j - \theta_i - \theta_{ij}$$, there is no dependency between these $$\mathbf{e}_{\theta}$$ and $$\mathbf{t}_i$$, rendering this Jacobian entry zero.


4. Fourth, consider
$$
\frac{\partial \mathbf{e}_{\theta}}{\partial \mathbf{x}_{i_{\theta}}}.
$$
The dependency between $$\mathbf{e}_{\theta} = \theta_j - \theta_i - \theta_{ij}$$ and $$\mathbf{x}_{i_{\theta}}$$ is clear -- it is a scaling by $$-1$$.

We place these four partial derivatives into a matrix $$A_{ij}$$:


$$
A_{ij} = \frac{\partial \mathbf{e}_{ij}}{\partial \mathbf{x}_i} = \begin{bmatrix}
-R_{ij}^T R_i^T & R_{ij}^T \frac{\partial R_{i}^T}{\partial \theta_i} (\mathbf{t}_j - \mathbf{t}_i) \\
\mathbf{0}^T & -1
\end{bmatrix}
$$

$$
B_{ij} = \frac{\partial \mathbf{e}_{ij}}{\partial \mathbf{x}_j} = \begin{bmatrix}
R_{ij}^T R_i^T & \mathbf{0} \\
\mathbf{0}^T & 1
\end{bmatrix}
$$


### Pose-Landmark Constraints

Consider the position of a landmark $$\mathbf{x}_l = \begin{bmatrix} x_l \\ y_l \end{bmatrix}$$:

$$
\mathbf{e}_{il} = R_i^T  (\mathbf{x}_l − \mathbf{t}_i) − \mathbf{z}_{il}
$$

The Jacobian with respect to $$\mathbf{e}_{il}$$ is a $$(3 \times 2)$$ matrix that we can express with two blocks:

$$
\frac{ \partial \mathbf{e}_{il} }{\partial \mathbf{x}_i} = \begin{bmatrix} \frac{\partial \mathbf{e}}{\partial \mathbf{t}_i} & \frac{\partial \mathbf{e}}{\partial \mathbf{x}_{i_{\theta}}} \end{bmatrix}
$$

$$
A = \frac{ \partial \mathbf{e}_{il} }{\partial \mathbf{x}_i} = \begin{bmatrix} -R_i^T & \frac{\partial R_i^T}{\partial \theta} (\mathbf{x}_l - \mathbf{t}_i) \end{bmatrix}
$$


We will shorten $$\mathbf{e}_{il}$$ to $$\mathbf{e}$$ for ease of notation. Although we could write out all 4 terms of this Jacobian, i.e.

$$
\frac{\partial \mathbf{e}_{il}}{\partial \mathbf{x}_l} = \frac{\partial \mathbf{e}}{\partial \mathbf{x}_l}  = \begin{bmatrix}
\frac{\partial \mathbf{e}_x }{ \partial l_x} & \frac{\partial \mathbf{e}_x }{ \partial l_y} \\
\frac{\partial \mathbf{e}_x }{ \partial l_x} & \frac{\partial \mathbf{e}_x }{ \partial l_y}
\end{bmatrix}
$$

we can simply notice that $$\mathbf{x}_l$$ is multiplied only by $$R_i^T$$ in the error function. Thus,

$$
B = \frac{\partial \mathbf{e}_{il}}{\partial \mathbf{x}_l} = \frac{\partial \mathbf{e}}{\partial \mathbf{x}_l} = R_i^T
$$










