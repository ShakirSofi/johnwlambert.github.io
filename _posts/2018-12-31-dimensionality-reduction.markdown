---
layout: post
title:  "Dimensionality Reduction"
permalink: /dimensionality-reduction/
excerpt: "PCA, ISOMAP, LLE, SNE, t-SNE "
mathjax: true
date:   2018-12-27 11:00:00
mathjax: true

---
Table of Contents:
- [A Basic SfM Pipeline](#sfmpipeline)
- [Cost Functions](#costfunctions)
- [Bundle Adjustment](#bundleadjustment)

<a name='sfmpipeline'></a>

## Linear Methods

### PCA


## Non-Linear Methods

## Creating Graphs from Geometric Point Data

Rely upon neighborhood relations. A graph can be constructed via (1) k-nearest neighbors or (2) $$\epsilon$$-balls.

### Isomap

### EigenMap

### Locally Linear Embedding (LLE)

LLE is a method that learns linear weights that locally reconstruct points in order to map points to a lower dimension [1,4]. The method requires solving two successive optimization problems and requires a connectivity graph. Almost all methods start with the nearest neihgbor graph, since it's the only thing that we can trust.

In the **first step of LLE**, we find weights that reconstruct each data point from its neighbors:

$$
\begin{array}{ll}
\underset{w}{\mbox{minimize }} & \| x_i - \sum\limits_{j \in N(i)} w_{ij}x_j \|^2 \\
\mbox{subject to} & \sum\limits_j w_{ij} = 1
\end{array}
$$

We can use linear least squares with Lagrange multipliers to obtain optimal linear combinations. 

In the **second step of LLE**, We then **fix these weights** $$w_{ij}$$ while optimizing for $$x_i^{\prime}$$. We try to find low-dimensional coordinates:

$$
\begin{array}{ll}
\underset{x_1^{\prime}, \dots, x_n^{\prime}}{\mbox{minimize }} \sum\limits_i \| x_i^{\prime} - \sum\limits_{j \in N(i)} w_{ij}x_j^{\prime} \|^2
\end{array}
$$

This is a sparse eigenvalue problem that requires constraints in order to prevent degenerate solutions:
- (1.) The coordinates $$x_i^{\prime}$$ can be translated by a constant
displacement without affecting the cost. We can remove this degree of freedom by requiring the coordinates to be centered on the origin. 
- (2.) We constrain the embedding vectors to have unit covariance.
The optimization problem becomes:

$$
\begin{array}{ll}
\underset{x_1^{\prime}, \dots, x_n^{\prime}}{\mbox{minimize }} & \sum\limits_i \| x_i^{\prime} - \sum\limits_{j \in N(i)} w_{ij}x_j^{\prime} \|^2 \\
\mbox{subject to} & \sum\limits_i x_i^{\prime} = 0 \\
& \frac{1}{n} \sum\limits_i x_i^{\prime}x_i^{\prime T} = I
\end{array}
$$

These weights $$w_{ij}$$ capture the local shape. As Roweis and Saul point out, "*LLE illustrates a general principle of manifold learning...that overlapping
local neighborhoods -- collectively analyzed -- can provide information about global
geometry*"" [4].

### Stochastic Neighbor Embedding (SNE)

The Stochastic Neighbor Embedding (SNE) converts high-dimensional points to low-dimensional points by preserving distances. The method takes a probabilistic point of view: high-dimensional Euclidean point distances are converted into conditional probabilities that represent similarities [2,3].


Similarity of datapoints in **high himension**: The conditional probability is given by

$$
p_{j \mid i} = \frac{\mbox{exp }\big( - \frac{\|x_i - x_j\|^2}{2 \sigma_i^2}\big) }{ \sum\limits_{k \neq i} \mbox{exp }\big( - \frac{\|x_i - x_k\|^2}{2 \sigma_i^2} \big)}
$$

Similarity of datapoints in **the low dimension**

$$
q_{j \mid i} = \frac{\mbox{exp }\big( - \|y_i - y_j\|^2\big) }{ \sum\limits_{k \neq i} \mbox{exp }\big( - \|y_i - y_k\|^2 \big)}
$$

If similarities between $$x_i,x_j$$ are correctly mapped to similarities between $$y_i,y_j$$ by SNE, then the conditional probabilities should be equal: $$q_{j \mid i} = p_{j \mid i}$$.

SNE seeks minimize the following cost function using gradient descent, which measures the dissimilarity between the two distributions (Kullback-Leibler divergence):

$$
C = \sum\limits_i KL(P_i || Q_i) = \sum\limits_i \sum\limits_j p_{j \mid i} \mbox{ log } \frac{p_{j \mid i}}{q_{j \mid i}}
$$

This is known as asymetric SNE. The gradient turns out to be analytically simple:

$$
\frac{\partial C}{\partial y_i} = 2 \sum\limits_j (p_{j \mid i} - q_{j \mid i} - p_{i \mid j} - q_{i \mid j} )(y_i - y_j)
$$

However, the Kullback-Leibler divergence is not symmetric, so a formulation with a joint distribution can be made.

### t-SNE

An improvement to SNE is t-Distributed Stochastic Neighbor Embedding (t-SNE). t-SNE employs a Gaussian in the high-dimension, but a t-Student distribution in low-dim. The t-Student distribution has longer tails than a Gaussian and is thus happier to have points far away than a Gaussian. The motivation for doing so is that in low-D, you have have less freedom than you would in the high-dimension to put many things closeby. This is because there is not much space around (crowded easily), so we penalize having points far away less.

The joint distribution in the low-distribution is:

$$
q_{ij} = \frac{ (1+ \| y_i −y_j \|^2)^{−1} }{ \sum\limits_{k \neq l} (1+ \|y_k −y_l\|^2)^{−1} }
$$

### MNIST examples


## References

[1] Leonidas Guibas. *Multi-Dimensional Scaling, Non-Linear Dimensionality Reduction*. Class lectures of CS233: Geometric and Topological Data Analysis, taught at Stanford University in 18 April 2018.

[2] Geoffrey Hinton and Sam Roweis. *Stochastic Neighbor Embedding*. Advances in Neural Information Processing Systems (NIPS) 2003, pages 857--864. [http://papers.nips.cc/paper/2276-stochastic-neighbor-embedding.pdf](http://papers.nips.cc/paper/2276-stochastic-neighbor-embedding.pdf).

[3] L.J.P. van der Maaten and G.E. Hinton. *Visualizing High-Dimensional Data Using t-SNE*. Journal of Machine Learning Research 9 (Nov):2579-2605, 2008.

[4] Sam T. Roweis and Lawrence K. Saul. *Nonlinear Dimensionality Reduction by Locally Linear Embedding*. Science Magazine, Vol. 290,  22 Dec. 2000.









