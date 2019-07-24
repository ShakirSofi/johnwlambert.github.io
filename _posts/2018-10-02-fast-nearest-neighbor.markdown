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
- [Brute Force](#rank)
- [Fast Affinity Matrices](#fastaffinitymatrices)
- [Speedup](#nullspace)


## The Nearest Neighbor Problem
Finding closest points in a high-dimensional space is a re-occurring problem in computer vision, especially when performing feature matching (e.g. with SIFT [1]) or computing Chamfer distances [2,3] for point set generation with deep networks.

Brute force methods can be prohibitively slow and much faster ways exist of computing with a bit of linear algebra.

## Nearest Neighbor Computation

Let $$\mathcal{A,B}$$ be sets. We are interested in the finding the nearest neighbor for each point in $$\mathcal{A}$$. Let $$a,b$$ be two points such that $$a \in \mathcal{A}$$, $$b \in \mathcal{B}$$. The nearest neighbor in $$\mathcal{B}$$ of a point $$a \in \mathcal{A}$$ is a point $$b \in \mathcal{B}$$, such that $$b = \mbox{arg} \underset{b \in \mathcal{B}}{\mbox{ min}} \|a-b\|_2$$. We can equivalently use the squared Euclidean distance $$\|a-b\|_2^2$$, since the square function is monotonically increasing for positive values, and distances are always positive. We will see that using the squared Euclidean distance to find the nearest neighbor will spare us some computation later.


The expression $$\|a-b\|_2^2$$ is equivalent to an inner product. It is equivalent to the Mahalonibis distance, $$(a-b)^TA(a-b)$$, when $$A=I$$, when working in Euclidean space (computing \\(\ell_2\\) norms):

Let $$a,b$$ be vectors, i.e. \\(a,b \in R^n\\) are vectors:

$$
\begin{aligned}
&= (a-b)^T(a-b) \\
&= (a^T-b^T)(a-b) \\
&= (a^T-b^T)(a-b) \\
&= a^Ta -b^Ta - a^Tb + b^Tb 
\end{aligned}
$$

Since $$-b^Ta$$ and  $$- a^Tb$$ are scalars (inner products), we can swap the order or arguments to find:

$$
\begin{aligned}
&= a^Ta -a^Tb - a^Tb + b^Tb \\
&= a^Ta -2 a^Tb + b^Tb
\end{aligned}
$$

Now we wish to compute these distances on all pairs of points in entire datasets simultaneously. We can form matrices \\(A \in R^{m_1 \times n}, B \in R^{m_2 \times n}\\) that hold our high-dimensional points.

Consider \\(AB^T\\):

$$
AB^T = \begin{bmatrix} - & a_1 & - \\ - & a_2 & - \\ - & a_3 & - \end{bmatrix} \begin{bmatrix} | & | & | \\ b_1^T & b_2^T & b_3^T \\ | & | & | \end{bmatrix}  
$$


<a name='extremaltrace'></a>
## Brute Force Nearest Neighbors

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

<a name='fastaffinitymatrices'></a>
## Fast Affinity Matrix Computation



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


We now demonstrate the 

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

## References

1. David Lowe. Distinctive Image Features
from Scale-Invariant Keypoints. IJCV, 2004. [PDF](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf).

2. H.  Fan,  H.  Su,  and  L.  J.  Guibas.   A  point  set  generationnetwork  for  3d  object  reconstruction  from  a  single  image. CVPR 2017. [PDF](https://arxiv.org/abs/1612.00603).

3. A. Kurenkov, J. Ji, A. Garg, V. Mehta, J. Gwak, C. B. Choy, and  S.  Savarese.   Deformnet:  Free-form  deformation  network for 3d shape reconstruction from a single image. WACV 2018. [PDF](https://arxiv.org/abs/1708.04672).


