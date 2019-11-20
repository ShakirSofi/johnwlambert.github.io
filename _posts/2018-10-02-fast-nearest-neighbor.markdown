---
layout: post
title:  "Fast Nearest Neighbors"
permalink: /fast-nearest-neighbor/
excerpt: "Vectorizing nearest neighbors (with no for-loops!)"
mathjax: true
date:   2018-10-02 11:00:00
mathjax: true

---
Table of Contents:


- [The Nearest Neighbor Problem](#nn-problem)
- [Nearest Neighbor Computation](#nn-computation)
- [Brute Force Nearest Neighbors](#brute-force)
- [Vectorized NN Derivation](#vectorized-nn)
- [Implementation: Fast Affinity Matrix Computation](#vectorized-numpy)
- [Speed Comparison](#speedup)



<a name='nn-problem'></a>
## The Nearest Neighbor Problem
Finding closest points in a high-dimensional space is a re-occurring problem in computer vision, especially when performing feature matching (e.g. with SIFT [1]) or computing Chamfer distances [2,3] for point set generation with deep networks. It is a cheaper way of finding point-to-point correspondences than optimal bipartite matching, as the Earth Mover's Distance requires.

Brute force methods can be prohibitively slow and much faster ways exist of computing with a bit of linear algebra.

<a name='nn-computation'></a>
## Nearest Neighbor Computation

Let $$\mathcal{A,B}$$ be sets. We are interested in the finding the nearest neighbor for each point in $$\mathcal{A}$$. Let $$a,b$$ be two points such that $$a \in \mathcal{A}$$, $$b \in \mathcal{B}$$. The nearest neighbor in $$\mathcal{B}$$ of a point $$a \in \mathcal{A}$$ is a point $$b \in \mathcal{B}$$, such that $$b = \mbox{arg} \underset{b \in \mathcal{B}}{\mbox{ min}} \|a-b\|_2$$. We can equivalently use the squared Euclidean distance $$\|a-b\|_2^2$$, since the square function is monotonically increasing for positive values, and distances are always positive. We will see that using the squared Euclidean distance to find the nearest neighbor will spare us some computation later.


The expression $$\|a-b\|_2^2$$ is equivalent to an inner product. It is equivalent to the Mahalanobis distance, $$(a-b)^TA(a-b)$$, when $$A=I$$, when working in Euclidean space (computing \\(\ell_2\\) norms):

Let $$a,b$$ be vectors, i.e. \\(a,b \in R^n\\) are vectors:

$$
\begin{aligned}
\|a-b\|_2^2 &= (a-b)^T(a-b) \\
&= (a^T-b^T)(a-b) \\
&= (a^T-b^T)(a-b) \\
&= a^Ta -b^Ta - a^Tb + b^Tb 
\end{aligned}
$$

Since $$-b^Ta$$ and  $$- a^Tb$$ are scalars (inner products), we can swap the order or arguments to find:

$$
\begin{aligned}
\|a-b\|_2^2 &= a^Ta -a^Tb - a^Tb + b^Tb \\
&= a^Ta -2 a^Tb + b^Tb
\end{aligned}
$$


<a name='brute-force'></a>
## Brute Force Nearest Neighbors

In the brute force regime, we would loop through all points $$a_i \in \mathcal{A}$$, and then loop through all points $$b_j \in \mathcal{B}$$, and find the distance $$\|a_i - b_j\|$$ with `np.linalg.norm(A[i] - B[j])`. This can be done with a double for-loop in Python:

```python
def naive_upper_triangular_compute_affinity_matrix(pts1, pts2):
    """
    Create an mxn matrix, where each (i,j) entry denotes
    the Mahalanobis distance between point i and point j,
    as defined by the metric "A". A has dimension (dim x dim).
    Use of a for loop makes this function somewhat slow.

    not symmetric
    """
    m1, n = pts1.shape

    # make sure feature vectors have the same length
    assert pts1.shape[1] == pts2.shape[1]
    m2, n = pts2.shape

    affinity_mat = np.zeros((m1,m2))
    for i in range(m1): # rows
        for j in range(m2): # cols
            diff = pts1[i] - pts2[j]
            norm = np.linalg.norm(diff)
            affinity_mat[i,j] = norm

    #affinity matrix contains the Mahalanobis distances
    return np.square(affinity_mat)
```
However, this method will be brutally slow for thousands, tens of thousands, or millions of points, which are quite common point cloud sizes in computer vision or robotics. We need a better way.

<a name='vectorized-nn'></a>
## Vectorized NN Derivation

Now we wish to compute these distances on all pairs of points in entire datasets simultaneously. We can form matrices \\(A \in R^{m_1 \times n}, B \in R^{m_2 \times n}\\) that hold our high-dimensional points.


We will see that nearest neighbor computation for all points boils down to only 3 required matrix products: $$AA^T, BB^T, AB^T$$.

Our goal is to find $$\|a_i - b_j\|_2^2 = a_i^Ta_i -2 a_i^Tb_j + b_j^Tb_j$$ for all $$i,j$$. We wish to build an affinity matrix $$D$$ such that the $$D_{ij}$$ entry contains the squared distance between $$a_i, b_j$$.

Consider a sets $$\mathcal{A,B}$$ with 3 points each. We form $$AA^T, BB^T$$:

$$
AA^T = \begin{bmatrix} - & a_1 & - \\ - & a_2 & - \\ - & a_3 & - \end{bmatrix} \begin{bmatrix} | & | & | \\ a_1^T & a_2^T & a_3^T \\ | & | & | \end{bmatrix}  =
\begin{bmatrix}
a_1^Ta_1 & a_1^T a_2 & a_1^T a_3 \\
a_2^Ta_1 & a_2^T a_2 & a_2^T a_3 \\
a_3^Ta_1 & a_3^T a_2 & a_3^T a_3
\end{bmatrix} 
$$

$$
BB^T = \begin{bmatrix} - & b_1 & - \\ - & b_2 & - \\ - & b_3 & - \end{bmatrix} \begin{bmatrix} | & | & | \\ b_1^T & b_2^T & b_3^T \\ | & | & | \end{bmatrix} = \begin{bmatrix}
b_1^T b_1 & b_1^T b_2 & b_1^T b_3 \\
b_2^T b_1 & b_2^T b_2 & b_2^T b_3 \\
b_3^T b_1 & b_3^T b_2 & b_3^T b_3
\end{bmatrix}  
$$

We are interested only in the diagonal elements $$a_i^Ta_i$$ and $$b_i^Tb_i$$. We will define $$T_A$$ and $$T_B$$ to contain tiled rows, where each row is a diagonal of $$AA^T$$ or $$BB^T$$, respectively:

$$
\begin{array}{ll}
T_A = \begin{bmatrix}
a_1^Ta_1 & a_2^Ta_2 & a_3^Ta_3 \\
a_1^Ta_1 & a_2^Ta_2 & a_3^Ta_3 \\
a_1^Ta_1 & a_2^Ta_2 & a_3^Ta_3 
\end{bmatrix}, & T_B = \begin{bmatrix}
b_1^T b_1 & b_2^T b_2 & b_3^T b_3 \\
b_1^T b_1 & b_2^T b_2 & b_3^T b_3 \\
b_1^T b_1 & b_2^T b_2 & b_3^T b_3 
\end{bmatrix}
\end{array}
$$

We now form $$AB^T$$:

$$
AB^T = \begin{bmatrix} - & a_1 & - \\ - & a_2 & - \\ - & a_3 & - \end{bmatrix} \begin{bmatrix} | & | & | \\ b_1^T & b_2^T & b_3^T \\ | & | & | \end{bmatrix} = \begin{bmatrix}
b_1^Ta_1 & b_2^Ta_1 & b_3^Ta_1 \\
b_1^Ta_2 & b_2^Ta_2 & b_3^Ta_2 \\
b_1^Ta_3 & b_2^Ta_3 & b_3^Ta_3 
\end{bmatrix}  
$$

Our desired affinity matrix $$D \in \mathbf{R}^{3 \times 3}$$ will contain entries $$D_{ij} = \|a_i - b_j\|_2^2$$:

$$
D = 
\begin{bmatrix} 
\| a_1 - b_1 \|_2^2 & \| a_1 - b_2 \|_2^2 & \|a_1 - b_3 \|_2^2 \\
\| a_2 - b_1 \|_2^2 & \| a_2 - b_2 \|_2^2 & \|a_2 - b_3 \|_2^2 \\
\| a_3 - b_1 \|_2^2 & \| a_3 - b_2 \|_2^2 & \|a_3 - b_3 \|_2^2
\end{bmatrix}
$$

In turns out that:

$$
\begin{aligned}
D &= T_A^T + T_B - 2 AB^T \\
D &= \begin{bmatrix}
a_1^Ta_1 & a_1^Ta_1 & a_1^Ta_1 \\
a_2^Ta_2 & a_2^Ta_2 & a_2^Ta_2 \\
a_3^Ta_3 & a_3^Ta_3 & a_3^Ta_3 
\end{bmatrix} + 
\begin{bmatrix}
b_1^T b_1 & b_2^T b_2 & b_3^T b_3 \\
b_1^T b_1 & b_2^T b_2 & b_3^T b_3 \\
b_1^T b_1 & b_2^T b_2 & b_3^T b_3 
\end{bmatrix} - 2 \begin{bmatrix}
b_1^Ta_1 & b_2^Ta_1 & b_3^Ta_1 \\
b_1^Ta_2 & b_2^Ta_2 & b_3^Ta_2 \\
b_1^Ta_3 & b_2^Ta_3 & b_3^Ta_3 
\end{bmatrix}  
\end{aligned}
$$

Since as you can see above, $$D_{ij} = \|a_i - b_j\|_2^2 = a_i^Ta_i -2 a_i^Tb_j + b_j^Tb_j$$ for all $$i,j$$.

<a name='vectorized-numpy'></a>
## Implementation: Fast Affinity Matrix Computation

The implementation requires just a few lines in Numpy:

```python
import numpy as np

def fast_affinity_mat_compute(X1,X2):
    """
    X is (m,n)
    A is (n,n)
    K is (m,m)
    """
    m1,n = X1.shape
    assert X1.shape[1] == X2.shape[1]
    m2,_ = X2.shape
    ab_T = np.matmul( X1, X2.T )
    a_sqr = np.diag(X1.dot(X1.T))
    b_sqr = np.diag(X2.dot(X2.T))
    a_sqr = np.tile( a_sqr, (m2,1) ).T
    b_sqr = np.tile( b_sqr, (m1,1) )
    return a_sqr + b_sqr - 2 * ab_T
```

<a name='speedup'></a>
## Speed Comparison

We now demonstrate the speedup of the vectorized approach over the brute force approach:

```python
def unit_test_arr_arr(m1,m2,n):
  """ """
  pts1 = np.random.rand(m1,n)
  pts2 = np.random.rand(m2,n)

  gt_aff = naive_upper_triangular_compute_affinity_matrix(pts1, pts2)

  pred_aff = fast_affinity_mat_compute(pts1,pts2)

  print(gt_aff)
  print(pred_aff)
  print(np.absolute(gt_aff - pred_aff).sum())


if __name__ == '__main__':

  m1 = 3 # 100
  m2 = 4 # 135
  n = 128

  np.random.seed(2)
  # unit_test_pt_arr(m1,m2,n)
  unit_test_arr_arr(m1,m2,n)
```

Now consider the speedup we've achieved:


## Alternative Method

```python
def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    source: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)
```


## References

1. David Lowe. Distinctive Image Features
from Scale-Invariant Keypoints. IJCV, 2004. [PDF](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf).

2. H.  Fan,  H.  Su,  and  L.  J.  Guibas.   A  point  set  generationnetwork  for  3d  object  reconstruction  from  a  single  image. CVPR 2017. [PDF](https://arxiv.org/abs/1612.00603).

3. A. Kurenkov, J. Ji, A. Garg, V. Mehta, J. Gwak, C. B. Choy, and  S.  Savarese.   Deformnet:  Free-form  deformation  network for 3d shape reconstruction from a single image. WACV 2018. [PDF](https://arxiv.org/abs/1708.04672).


