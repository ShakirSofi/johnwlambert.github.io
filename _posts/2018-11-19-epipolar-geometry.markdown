---
layout: post
title:  "Epipolar Geometry"
permalink: /epipolar-geometry/
excerpt: "The Fundamental matrix, solution via SVD."
mathjax: true
date:   2018-11-19 11:01:00
mathjax: true

---

## Why know Epipolar Geometry?
Modern robotic computer vision tasks like Structure from Motion (SfM) and Simultaneous Localization and Mapping (SLAM) would not be possible without feature matching. Tools from Epipolar Geometry are an easy way to discard outliers in feature matching and are widely used.


## The Fundamental Matrix


The Fundamental matrix provides a correspondence \\(x^TFx^{\prime} = 0\\), where \\(x^{\prime},\\) are 2D corresponding points in separate images. In other words,

$$
\begin{bmatrix} u^{\prime} & v^{\prime} & 1 \end{bmatrix} \begin{bmatrix} f_{11} & f_{12} & f_{13} \\ f_{21} & f_{22} & f_{23} \\ f_{31} & f_{32} & f_{33} \end{bmatrix} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = 0
$$

Longuet-Higgins' 8-Point Algorithm [1] provides the solution for estimating \\(F\\) if at least 8 point correspondences are provided. A system of linear equations is formed as follows:

$$
    Af = \begin{bmatrix} u_1 u_1^{\prime} & u_1v_1^{\prime} & u_1 & v_1 u_1^{\prime} & v_1 v_1^{\prime} & v_1 & u_1^{\prime} & v_1^{\prime} & 1 \\ \vdots & \vdots  & \vdots  & \vdots  & \vdots  & \vdots  & \vdots  & \vdots & \vdots  \\   u_n u_n^{\prime} & u_n v_n^{\prime} & u_n & v_n u_n^{\prime} & v_n v_n^{\prime} & v_n & u_n^{\prime} & v_n^{\prime} & 1 \end{bmatrix} \begin{bmatrix} f_{11} \\ f_{12} \\ f_{13} \\ f_{21} \\ \vdots \\ f_{33} \end{bmatrix} = \begin{bmatrix} 0 \\ \vdots \\ 0 \end{bmatrix}
$$

The matrix-vector product above can be driven to zero by minimizing the norm, and avoiding the degenerate solution that \\(x=0\\) with a constraint that the solution lies upon the unit ball, e.g.

$$
\begin{array}{ll}
  \underset{\|x\|=1}{\mbox{minimize}} & \|A x \|_2^2 = x^TA^TAx = x^TBx
  \end{array}
$$

By the Courant-Fisher characterization, it is well known that if \\(B\\) is a \\(n \times n\\) symmetric matrix with eigenvalues \\(\lambda_1 \geq \lambda_2 \geq \dots \geq \lambda_n \\) and corresponding eigenvectors \\(v_1, \dots, v_n\\), then

$$
    v_n = \mbox{arg } \underset{\|x\|=1}{\mbox{min }} x^TBx
$$

meaning the eigenvector associated with the smallest eigenvalue \\(\lambda_n\\) of \\(B\\) is the solution \\(x^{\star}\\). The vector \\(x^{\star}\\) contains the 9 entries of the Fundamental matrix \\(F^{\star}\\).

This is a specific instance of the extremal trace \\([2]\\) (or trace minimization on a unit sphere) problem, with \\(k=1\\), i.e.
\begin{equation}
    \begin{array}{ll}
    \mbox{minimize} & \mathbf{\mbox{Tr}}(X^TBX) \\
    \mbox{subject to} & X^TX=I_k
    \end{array}
\end{equation}
where \\(I_k\\) denotes the \\(k \times k\\) identity matrix. The unit ball constraint avoids the trivial solution when all eigenvalues \\(\lambda_i\\) are zero instead of a single zero eigenvalue.

In our case, since $U,V$ are orthogonal matrices (with orthonormal columns), then \\(U^TU=I\\). Thus, the SVD of \\(B\\) yields 
\begin{equation}
A^TA = (U\Sigma V^T)^T (U\Sigma V^T) = V\Sigma U^TU \Sigma V^T = V \Sigma^2 V^T.
\end{equation}

Since \\(B=A^TA\\), \\(B\\) is symmetric, and thus the columns of \\(V=\begin{bmatrix}v_1 \dots v_n \end{bmatrix}\\) are eigenvectors of \\(B\\). \\(V\\) can equivalently be computed with the SVD of \\(A\\) or \\(B\\), since \\(V\\) appeares in both decompositions: \\(A=U \Sigma V^T\\) and \\(B=V\Sigma^2V^T\\).
\\ \\

## Proof: the SVD provides the solution
 The proof is almost always taken for granted, but we will provide it here for completeness. Because \\(B\\) is symmetric, there exists a set of \\(n\\) orthonormal eigenvectors, yielding an eigendecomposition \\(B=V^T \Lambda V\\). Thus,

$$
\begin{array}{llll}
    \mbox{arg } \underset{\|x\|=1}{\mbox{min }} & x^TBx = \mbox{arg } \underset{\|x\|=1}{\mbox{min }} & x^TV^T \Lambda Vx = \mbox{arg } \underset{\|x\|=1}{\mbox{min }} & (Vx)^T \Lambda (Vx)
\end{array}
$$

Since \\(V\\) is orthogonal, \\(\|Vx\| = \|x\|\\), thus minimizing \\((Vx)^T \Lambda (Vx)\\) is equivalent to minimizing \\(x^T \Lambda x\\). Since \\(\Lambda\\) is diagonal, \\(x^TBx = \sum\limits_{i=1}^{n} \lambda_i x_i^2\\) where \\(\{\lambda_i\}_{i=1}^n\\) are the eigenvalues of \\(B\\). Let \\(q_i=x_i^2\\), meaning \\(q_i\geq 0\\) since it represents a squared quantity. Since \\(\|x\|=1\\), then \\(\sqrt{\sum\limits_i x_i^2}=1\\), \\(\sum\limits_i x_i^2=1 \\), \\(\sum\limits_i q_i = 1\\). Thus, 

\begin{equation}
 \underset{\|x\|=1}{\mbox{min }}  x^TBx= \underset{\|x\|=1}{\mbox{min }} \sum\limits_{i=1}^{n} \lambda_i x_i^2= \underset{q_i}{\mbox{min }} \sum\limits_i \lambda_i q_i = \underset{q_i}{\mbox{min }} \lambda_n \sum\limits_i q_i = \lambda_n
\end{equation}

where \\(\lambda_n\\) is the smallest eigenvalue of \\(B\\). The last line follows since \\(q_i \geq 0\\) and \\(\sum\limits_i q_i = 1\\), therefore we have a convex combination of a set of numbers \\( \{\lambda_i\}_{i=1}^n \\) on the real line. By properties of a convex combination, the result must lie in between the smallest and largest number. Now that we know the minimum is \\(\lambda_n\\), we can obtain the argmin by the following observation:

If \\(v\\) is an eigenvector of \\(B\\), then 
\begin{equation}
    Bv = \lambda_n v
\end{equation}
Left multiplication with \\(v^T\\) simplifies the right side because \\(v^Tv=1\\), by our constraint that \\(\|x\|=1\\). We find:
\begin{equation}
    v^T(Bv) = v^T (\lambda_n v) = v^Tv \lambda_n = \lambda_n
\end{equation}
Thus the eigenvector \\(v\\) associated with the eigenvalue \\(\lambda_n\\) is \\(x^{\star}\\). \\( \square\\)



References:

[1] H. Longuet Higgins. A computer algorithm for reconstructing a scene from two projections. *Nature*, 293, 1981.

[2] S.  Boyd.   Low  rank  approximation  and  extremal  gain  problems,  2008.   URL [http://ee263.stanford.edu/notes/low_rank_approx.pdf](http://ee263.stanford.edu/notes/low_rank_approx.pdf).

