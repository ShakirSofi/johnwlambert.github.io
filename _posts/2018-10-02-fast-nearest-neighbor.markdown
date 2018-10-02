---
layout: post
title:  "Fast Nearest Neighbors"
permalink: /fast-nearest-neighbor/
mathjax: true
date:   2018-10-02 11:00:00
mathjax: true

---
Table of Contents:
- [Brute Force](#rank)
- [Fast Affinity Matrices](#fastaffinitymatrices)
- [Speedup](#nullspace)


## Nearest Neighbor
Finding closest points in a high-dimensional space is a re-occurring problem in computer vision, especially when performing feature matching (e.g. with [SIFT](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)).

Brute force methods can be prohibitively slow and much faster ways exist of computing with a bit of linear algebra.

We are interested in the quantity


Since \\(a,b \in R^n\\) are vectors, we can expand the Mahalanobis distance. When \\(A=I\\), we are working in Euclidean space (computing \\(\ell_2\\) norms):

$$
=(a-b)^TA(a-b)
$$

$$
=(a^T-b^T)A(a-b)
$$

$$
=(a^TA-b^TA)(a-b)
$$

$$
=a^TAa-b^TAa - a^TAa + b^TAb
$$

Now we wish to compute these on entire datasets simultaneously. We can form matrices \\(A \in R^{m_1 \times n}, B \in R^{m_2 \times n}\\) that hold our high-dimensional points.

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




