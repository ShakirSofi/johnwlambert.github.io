---
layout: post
title:  "Dimensionality Reduction"
permalink: /dimensionality-reduction/
excerpt: "PCA, geodesic distances, ISOMAP, LLE, SNE, t-SNE "
mathjax: true
date:   2018-12-27 11:00:00
mathjax: true

---
Table of Contents:
- [Need for Dimensionality Reduction](#need-for-dr)
- [PCA](#pca)
- [Nonlinear Dimensionality Reduction Methods](#nonlinear-dr-methods)
- [ISOMAP](#isomap)
- [LLE](#lle)
- [SNE](#sne)
- [t-SNE](#t-sne)

<a name='need-for-dr'></a>

## The Need for Dimensionality Reduction

When the dimensionality of data points is high, interpretation or visualization of the data can be quite challenging.

<a name='pca'></a>

### Linear Methods

## Principal Components Analysis (PCA)

PCA is an algorithm for linear dimensionality reduction that can be surprisingly confusing. There are many different ways to explain the algorithm, and it is not new -- Karl Pearson (University College, London) introduced the ideas in 1901 and Harold Hostelling (Columbia) developed the terminology and more of the math about principal components in 1933.

To understand PCA, we must first consider data as points in a Euclidean space.  Suppose the lies in a high-dimensional subspace, but the real space of the data could be low-dimensional. We would like to identify the real, lower-dim subspace where lower-dim structure corresponds to linear subspaces in the high-dim space.

**PCA has one key hyperparameter: the dimensionality of the lower-dim space.** How many dimensions do we need to explain most of the variability in the data?

The key idea is that we will project the points from high-D to low-D, and then when we project them back into high-D, the "reconstruction error" should be minimized.



\item Pearson (1901) -- find a single lower dimensional subspace that captures most of the variation in the data
\item What should be the dimensionality of the lower-dim space? 

Acts as a proxy for the real space

Minimize errors introudced by projecting the data into this subspace

Assumption: data has zero mean, ``centered data''

Move the origin to the middle of the data, the data will live in the hyperplane. Becomes subspace in new translated coordinate system

As a consequence, variance becomes just the 2nd moment

Data points in 2D -> project onto the best fit line (projection onto 1D), then assess how well the data looks now when you lay out the projected points onto the best fit line (this is the reconstruction)
Bring it back onto the best fit line (multiply by vector)

$$
x_{||} = vv^Tx
$$

Minimize an error function: error equivalent to maximizing the variance of the projected data

$$
E = \frac{1}{M} \sum\limits_I \sum\limits_M ... = < \|x^{\prime \mu} - x_{||}^{\prime \mu}\|^2 >_{\mu}
$$

$I$ dimensions
WHY IS THIS?
Trying to find the directions of maximum variance
Knowing the covariance matrix tells us about the 2nd order moments, but not about the highest-order structure of the data (unless data is normal)

If the covariance matrix is diagonal, then finding the direction of maximum variance is easy

Rotate your coordinate system so that the covariance matrix becomes diagonal.

This becomes an eigenvalue problem: the eigenvectors of the covariance matrix point in the directions of maximal variance?

Solve all top-k possible subspaces

Valid for all of them

Check the error depending on many dimensions we used. A tradeoff
$\mathbf{V}^T \mathbf{V} = I$ because $\mathbf{V}$ is an orthonormal matrix
Projection $x_{||} = \mathbf{V}y = \mathbf{V}\mathbf{V}^Tx$

Projection operator = $\mathbf{V}\mathbf{V}^T$

But in general $P$ is of low rank and loses info

Variance and reconstruction error

$$
<x^Tx> - <y_{||}^T y_{||}>
$$

Want to be able to maximize the variance in the projected subspace... (FIND OUT WHY THAT IS)

Spectral analysis of the covariance matrix:
if covariance matrix is symmetric, it always has real eigenvalues and orthogonal eigenvectors

The total variance fo the data is just the sum of the eigenvalues of the covariance matrix, all non-negative

Have diagonalized the covariance matrix, aligned, which was what we set out to do

This is an extremal trace problem?

$$
\sum\limits_i \lambda_i \sum\limits_p (v_{ip}^{\prime})^2
$$

$$ = \mbox{tr }(V^{\prime T} \Lambda V^{\prime})$$

To maximize the variance, need to put as mucch weight as possible on the large eigenvalues

The reconstruction error is 

$$
\sum\limits_{i=P+1}^I \lambda_i
$$

Aligns the axis with your data, so that for any $P$, can find $P$-dimensional subspace

### Eigenvectors Solve PCA

**Main Lesson from PCA**: For any P, the optimal $P$-dimensional subspace is the one spanned by the first $P$ eigenvectors of the covariance matrix

The eigenvectors give us the frame that we need -- align with it

### PCA Examples

First principal component does not always have semantic meaning (combination of arts, recreation, transportation, health, and housing explain the ratings of places the most)

This is simply what the analysis gives.

The math is beautiful, but semantic meaning is not always clear. Sparse PCA tries to fix this (linear combinations of only a small number of variables)

Geography of where an individual came from is reflected in their genetic material

Machinery gives us a way to understand distortions of changes: a recipe encoded as a matrix

PCA works very well here

View matrices as points in high-dimensional space

Recover grid structure for deformed sphere

Get circular pattern of galloping horse from points

### SVD Trick

What if we have fewer data points than dimensions $$M<I$$? Think of images...

By SVD, transpose your data

Think of your data as rows, not columns

Data becomes first pixel across 1000 images

Covariance matrix $$C_2$$ of transposed problem

Rows of transformed X (PCA'd X) are just eigenvectors of $$C_1$$

Corresponding eigenvalues are just those of $$C_2$$, scaled by $$I/M$$


ntroduced by Pearson (1901)
and Hotelling (1933) to
describe the variation in a set
of multivariate data in terms of
a set of uncorrelated variables.
PCA looks for a single lower
dimensional subspace that
captures most of the variation
in the data.
Specifically, we aim to
minimize the error introduced
by projecting the data into this
linear subspace.

Use spectral analysis of the covariance matrix C of the
data
For any integer p, the error-minimizing p-dimensional
subspace is the one spanned by the first p eigenvectors
of the covariance matrix


Eigenfaces (PCA on face images)

M. Turk and A. Pentland, Face Recognition using Eigenfaces, CVPR 1991


https://www.stat.cmu.edu/~cshalizi/uADA/12/lectures/ch18.pdf

http://theory.stanford.edu/~tim/s17/l/l8.pdf



## Multi-Dimensional Scaling (MDS)


<a name='nonlinear-dr-methods'></a>

## Non-Linear Methods

Many data sets contain essential nonlinear structures and unfortunately these structures are invisible to linear methods like PCA [1].

For example, Euclidean distance between points cannot disentangle or capture the structure of manifolds like the "swiss roll." Instead, we need geodesic distance: distance directly on the manifold.

Furthermore, PCA cares about large distances. The goal is to maximize variance, and variance comes from things that are far apart. If you care about small distances, PCA is the wrong tool to use.

## Creating Graphs from Geometric Point Data

Rely upon neighborhood relations. A graph can be constructed via (1) k-nearest neighbors or (2) $$\epsilon$$-balls.

<a name='isomap'></a>

### Isometric Feature Mapping (ISOMAP)

In the case of the swiss roll, Isometric Feature Mapping (ISOMAP) produces an unrolled, planar version of data that respects distances [5]. This is "isometric" unrolling.

The ISOMAP algorithm involves three main steps, as outlined by Guibas [1]:

- (1.) Form a nearest-neighbor graph $$\mathcal{G}$$ on the original data points, weighing the edges by their original distances $$d_x(i,j)$$. One can build the NN-graph either (A) with a fixed radius/threshold $$\epsilon$$, or by using a (B) fixed \# of neighbors.
- (2.) Estimate the geodesic distances $$d_{\mathcal{G}}(i,j) $$ between all pairs of points on the sampled manifold by computing their shortest path distances in the graph $$\mathcal{G}$$. This can be done with classic graph algorithms for all-pairs shortest-path algorithm (APSP)s: Floyd/Warshall's algorithm or Dijkstra's algorithm. Initially, all pairs given distance $$\infty$$, except for neighbors, which are connected.
- (3.) Construct an embedding of the data in $$d$$-dimensional Euclidean space that best preserves the inter-point distances $$d_{\mathcal{G}}(i,j) $$. This is performed via Multi-Dimensional Scaling (MDS).

ISOMAP actually comes with recovery guarantees that discovered structure equal to actual structure of the manifold, especially as the graph point density increases. MDS and ISOMAP both converge, but ISOMAP gets there more quickly.


<a name='lle'></a>

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

<a name='sne'></a>

### Stochastic Neighbor Embedding (SNE)

The Stochastic Neighbor Embedding (SNE) converts high-dimensional points to low-dimensional points by preserving distances. The method takes a probabilistic point of view: high-dimensional Euclidean point distances are converted into conditional probabilities that represent similarities [2,3].


Similarity of datapoints in **high himension**: The conditional probability is given by

$$
p_{j \mid i} = \frac{\mbox{exp }\big( - \frac{\|x_i - x_j\|^2}{2 \sigma_i^2}\big) }{ \sum\limits_{k \neq i} \mbox{exp }\big( - \frac{\|x_i - x_k\|^2}{2 \sigma_i^2} \big)}
$$

Similarity of datapoints in **the low dimension**:

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

<a name='t-sne'></a>

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

[5] J. B. Tenenbaum, V. de Silva and J. C. Langford. *A Global Geometric Framework for Nonlinear Dimensionality Reduction*. Science 290 (5500): 2319-2323, 22 December 2000.

[6] Laurenz Wiskott. *Principal Component Analysis*. 11 March 2004. [Online PDF](https://pdfs.semanticscholar.org/d657/68e1dad46bbdb5cfb17eb19eb07cc0f5947c.pdf).

[7] Karl Pearson. *On Lines and Planes of Closest Fit to Systems of Points in Space*. 1901. Philosophical Magazine. 2 (11): 559–572.

[8] H Hotelling. *Analysis of a complex of statistical variables into principal components*. 1933. Journal of Educational Psychology, 24, 417–441, and 498–520.
Hotelling, H (1936). "Relations between two sets of variates". Biometrika. 28 (3/4): 321–377. doi:10.2307/2333955. JSTOR 2333955.



