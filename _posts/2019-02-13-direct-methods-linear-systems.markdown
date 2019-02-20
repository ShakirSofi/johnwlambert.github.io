---
layout: post
title:  "Direct Methods for Linear System Solving"
permalink: /direct-methods/
excerpt: "LU, Cholesky, QR"
mathjax: true
date:   2019-02-12 11:00:00
mathjax: true

---
Table of Contents:
- [Back Substitution](#back-substitution)
- [LU Factorization](#lu)
- [Cholesky Factorization](#cholesky)
- [QR Factorization](#qr)


Suppose we wish to solve a problem $$Ax=b$$, where $$A,b$$ are known. Each row of $$A$$ and $$b$$ represents a linear equation, so the problem represents a system of linear equations. This is one of the most well-studied problems in mathematics. It turns out that there are *direct methods* and *iterative methods* to solve this type of problem. Iterative methods are best when $$A$$ is sparse and very large. 

The problem $$Ax=b$$ is easy in two particular cases:

1. When $$A$$ is diagonal.
2. When $$A$$ is triangular.

We will focus on the case when $$A$$ is triangular. Direct methods focus on creating a triangular decomposition of the matrix $$A$$ such that the system can be quickly and easily solved in $$n^2$$ steps by a process called *back-substitution*, which we will describe in detail. Examples of such factorizations are the LU, Cholesky, and QR factorizations.

However, direct methods (i.e. computing such factorizations) are too computationally costly to solve on a desktop machine when $$A$$ has tens or hundreds of thousands of rows. You can verify this yourself quite easily
```python
import numpy as np
n = 100000
A = np.random.randint(-100,100,(n,n))
np.linalg.qr(A)
```
This should hang indefinitely and you will mostly like have to kill the Terminal window to prevent your computer from freezing up. However, for smaller matrices, direct methods are an easy and efficient approach. We will explain backsubstitution and then focus on 3 direct methods in this tutorial.

<a name='back-substitution'></a>

## Back Substitution for Triangular Matrices

Consider the system $$Ax=b$$:

$$
\begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} = \begin{bmatrix} b_1 \\ b_2 \\ b_3 \end{bmatrix}
$$

In general, the $$k$$-th equation is

$$
0+ \cdots + 0 + a_{kk}x_k + a_{k,k+1}x_{k+1} +···+a_{k,n}x_n =b_k
$$

Thus, by rearranging terms we can always solve for $$x_k$$ as follows:

$$
x_k=(b_k− \sum\limits_{j=k+1}^{n}a_{kj}x_j)/a_{kk} 
$$

```python
def back_substitution(A,b):
	""" """
	n,_ = A.shape
	x = np.zeros((n,1))
	for k in range(n-1,-1,-1):
		for j in range(k+1,n):
			b[k] = b[k] - A[k,j] * x[j]
		x[k] = b[k] / A[k,k]

	return x
```


```python
def back_substitution_demo():
	n = 10
	A = np.random.randint(1,10,(n,n))
	A = np.triu(A)
	x = np.random.randint(1,10,(n,1))
	b = A.dot(x)

	x_est = back_substitution(A,b)
```

<a name='lu'></a>

## LU Decomposition
In the LU Decomposition/Factorization, we seek to find two matrices $$L,U$$ such that $$A = L * U$$, and where $$L$$ is unit lower triangular (1's along the diagonal), and where $$U$$ is upper triangular.


```python
def LU_demo():
	""" """
	n = 10
	A = np.random.randint(1,10,(n,n))

	L,U = LU(A)
```

$$ 
A =  \begin{bmatrix}  7& 5& 4& 6& 7& 1& 4& 1& 1& 2 \\
            9& 1& 2& 2& 4& 8& 9& 5& 4& 5 \\
            6& 5& 6& 2& 1& 5& 6& 2& 7& 4 \\
            6& 8& 3& 6& 2& 5& 8& 4& 7& 3 \\
            6& 7& 6& 7& 8& 4& 8& 7& 8& 8 \\
            4& 4& 4& 4& 5& 2& 1& 7& 4& 2 \\
            3& 7& 4& 9& 7& 5& 3& 8& 2& 3 \\
            7& 1& 8& 8& 7& 6& 4& 8& 5& 8 \\
            4& 5& 3& 5& 1& 4& 6& 4& 3& 3 \\
            3& 3& 3& 3& 7& 4& 5& 2& 5& 9 \end{bmatrix}, M_0 = \begin{bmatrix} 1.     &  0.        &  0.        &  0.        &  0.        & 0.        &  0.        &  0.        &  0.        &  0.        \\
                -1.285 &  1.        &  0.        &  0.        &  0.        & 0.        &  0.        &  0.        &  0.        &  0.        \\
                -0.857 &  0.        &  1.        &  0.        &  0.        & 0.        &  0.        &  0.        &  0.        &  0.        \\
                -0.857 &  0.        &  0.        &  1.        &  0.        & 0.        &  0.        &  0.        &  0.        &  0.        \\
                -0.857 &  0.        &  0.        &  0.        &  1.        & 0.        &  0.        &  0.        &  0.        &  0.        \\
                -0.571 &  0.        &  0.        &  0.        &  0.        & 1.        &  0.        &  0.        &  0.        &  0.        \\
                -0.428 &  0.        &  0.        &  0.        &  0.        & 0.        &  1.        &  0.        &  0.        &  0.        \\
                -1.    &  0.        &  0.        &  0.        &  0.        & 0.        &  0.        &  1.        &  0.        &  0.        \\
                -0.571 &  0.        &  0.        &  0.        &  0.        & 0.        &  0.        &  0.        &  1.        &  0.        \\
                -0.428 &  0.        &  0.        &  0.        &  0.        & 0.        &  0.        &  0.        &  0.        &  1.       \\
\end{bmatrix}
$$


```python
def LU(A):
	""" 
	Alternatively, we could compute multipliers,
	update the A matrix entries that change 
	(lower square block), and then store the
	multipliers in the empty, lower triangular part of A.
	"""
	n,_ = A.shape
	M = np.zeros((n-1,n,n))
	U = A.copy()
	# loop over the columns of U
	for k in range(n-1):
		M[k] = np.eye(n)
		# compute multipliers for each row under the diagonal
		for i in range(k+1,n):
			M[k,i,k] = -U[i,k] / U[k,k]
		# must update the matrix to compute
		# multipliers for next column
		pdb.set_trace()
		U = M[k].dot(U)

	L = np.eye(n)
	# left-multiply higher M matrices
	for k in range(n-1):
		L = M[k].dot(L)
	L = np.linalg.inv(L)
	return L,U
```

Suppose we wish to see how $$A$$ changes when $$M_0$$ is applied: the zero'th column is now upper triangular.

$$
M_0 A = \begin{bmatrix} 7. &  5. &  4. &  6. &  7. &  1. &  4. &  1. &  1. &  2.  \\
			 0. & -5.4& -3.1& -5.7& -5. &  6.7&  3.9&  3.7&  2.7&  2.4 \\
			 0. &  0.7&  2.6& -3.1& -5. &  4.1&  2.6&  1.1&  6.1&  2.3 \\
			 0. &  3.7& -0.4&  0.9& -4. &  4.1&  4.6&  3.1&  6.1&  1.3 \\
			 0. &  2.7&  2.6&  1.9&  2. &  3.1&  4.6&  6.1&  7.1&  6.3 \\
			 0. &  1.1&  1.7&  0.6&  1. &  1.4& -1.3&  6.4&  3.4&  0.9 \\
			 0. &  4.9&  2.3&  6.4&  4. &  4.6&  1.3&  7.6&  1.6&  2.1 \\
			 0. & -4. &  4. &  2. &  0. &  5. &  0. &  7. &  4. &  6.  \\
			 0. &  2.1&  0.7&  1.6& -3. &  3.4&  3.7&  3.4&  2.4&  1.9 \\
			 0. &  0.9&  1.3&  0.4&  4. &  3.6&  3.3&  1.6&  4.6&  8.1 \end{bmatrix}, A = \begin{bmatrix}  7& 5& 4& 6& 7& 1& 4& 1& 1& 2 \\
            9& 1& 2& 2& 4& 8& 9& 5& 4& 5 \\
            6& 5& 6& 2& 1& 5& 6& 2& 7& 4 \\
            6& 8& 3& 6& 2& 5& 8& 4& 7& 3 \\
            6& 7& 6& 7& 8& 4& 8& 7& 8& 8 \\
            4& 4& 4& 4& 5& 2& 1& 7& 4& 2 \\
            3& 7& 4& 9& 7& 5& 3& 8& 2& 3 \\
            7& 1& 8& 8& 7& 6& 4& 8& 5& 8 \\
            4& 5& 3& 5& 1& 4& 6& 4& 3& 3 \\
            3& 3& 3& 3& 7& 4& 5& 2& 5& 9 \end{bmatrix}
$$
I'm actually rounding the entries above, i.e. showing the output of:
```python
np.round(M[0].dot(A), 1)
```


A fact that makes computing the inverses of these elementary GE matrices trivial: simply negate the spike entries below the diagonal.

$$
M_0 = \begin{bmatrix} 1.     &  0.        &  0.        &  0.        &  0.        & 0.        &  0.        &  0.        &  0.        &  0.        \\
                -1.285 &  1.        &  0.        &  0.        &  0.        & 0.        &  0.        &  0.        &  0.        &  0.        \\
                -0.857 &  0.        &  1.        &  0.        &  0.        & 0.        &  0.        &  0.        &  0.        &  0.        \\
                -0.857 &  0.        &  0.        &  1.        &  0.        & 0.        &  0.        &  0.        &  0.        &  0.        \\
                -0.857 &  0.        &  0.        &  0.        &  1.        & 0.        &  0.        &  0.        &  0.        &  0.        \\
                -0.571 &  0.        &  0.        &  0.        &  0.        & 1.        &  0.        &  0.        &  0.        &  0.        \\
                -0.428 &  0.        &  0.        &  0.        &  0.        & 0.        &  1.        &  0.        &  0.        &  0.        \\
                -1.    &  0.        &  0.        &  0.        &  0.        & 0.        &  0.        &  1.        &  0.        &  0.        \\
                -0.571 &  0.        &  0.        &  0.        &  0.        & 0.        &  0.        &  0.        &  1.        &  0.        \\
                -0.428 &  0.        &  0.        &  0.        &  0.        & 0.        &  0.        &  0.        &  0.        &  1.       \\
\end{bmatrix},  M_0^{-1} = \begin{bmatrix} 1.     &  0.        &  0.        &  0.        &  0.        & 0.        &  0.        &  0.        &  0.        &  0.        \\
                1.285 &  1.        &  0.        &  0.        &  0.        & 0.        &  0.        &  0.        &  0.        &  0.        \\
                0.857 &  0.        &  1.        &  0.        &  0.        & 0.        &  0.        &  0.        &  0.        &  0.        \\
                0.857 &  0.        &  0.        &  1.        &  0.        & 0.        &  0.        &  0.        &  0.        &  0.        \\
                0.857 &  0.        &  0.        &  0.        &  1.        & 0.        &  0.        &  0.        &  0.        &  0.        \\
                0.571 &  0.        &  0.        &  0.        &  0.        & 1.        &  0.        &  0.        &  0.        &  0.        \\
                0.428 &  0.        &  0.        &  0.        &  0.        & 0.        &  1.        &  0.        &  0.        &  0.        \\
                1.    &  0.        &  0.        &  0.        &  0.        & 0.        &  0.        &  1.        &  0.        &  0.        \\
                0.571 &  0.        &  0.        &  0.        &  0.        & 0.        &  0.        &  0.        &  1.        &  0.        \\
                0.428 &  0.        &  0.        &  0.        &  0.        & 0.        &  0.        &  0.        &  0.        &  1.       \\
\end{bmatrix}
$$
Indeed, $$L$$ is lower-triangular and $$U$$ is upper-triangular, as desired:

$$
L = \begin{bmatrix}   1.&    0.&   0.&    0.&   0.&    0.&    0.&    0.&   0. & 0. \\
    1.&    1.&    0.&   0.&    0.&    0.&    0.&    0.&   0.& 0. \\
    1.&   0.&    1.&   0.&   0.&   0.&    0.&    0.&    0.& 0. \\
    1.&   -1.&   -1.&    1.&   0.&    0.&   0.&   0.&   0.& 0. \\
    1.&   0.&    0.&   0.&    1.&   0.&    0.&   0.&    0. & 0. \\
    1.&   0.&    0.&   0.&    1.&    1.&    0.&    0.&    0.& 0. \\
    0.&   -1.&   -0.&   0.&   -4.&  -69.&    1.&   0.&   0. & 0. \\
    1.&    1.&    3.&   -2.&  -19.& -249.&    4.&    1.&   0. & 0. \\
    1.&   0.&   0.&    0.&   -5.&  -67.&    1.&    0.&    1.& 0. \\
    0.&   0.&    0.&   -0.&    6.&   53.&   -1.&   0.&   -1.& 1. \end{bmatrix} , U =  \begin{bmatrix}  7. &    5. &    4. &    6. &    7. &    1. &    4. &    1. &  1. &    2.  \\
   0. &   -5.4&   -3.1&   -5.7&   -5. &    6.7&    3.9&    3.7&  2.7&    2.4 \\
   0. &    0. &    2.2&   -3.9&   -5.7&    5. &    3.1&    1.6&   6.5&    2.6 \\
   0. &    0. &    0. &   -7.7&  -14.2&   14.7&   10.9&    7.6&   15.8&    6.1 \\
   0. &    0. &    0. &    0. &    0.6&    5.7&    6.2&    8. &  7.1&    6.9 \\
   0. &    0. &    0. &    0. &    0. &   -0.5&   -3.8&    3. &  -0.7&   -2.9 \\
   0. &    0. &    0. &    0. &    0. &    0. & -230.1&  247.9& -15.8& -169.  \\
   0. &    0. &    0. &    0. &    0. &    0. &    0. &   33.3& 27.7&    8.8 \\
   0. &    0. &    0. &    0. &    0. &    0. &    0. &    0. & -3.7&   -0.8 \\
   0. &    0. &    0. &    0. &    0. &    0. &    0. &    0. & 0. &    3.   \end{bmatrix}
$$

There is a very small amount of numerical error introduced in the process:

$$ \sum\limits_{i,j} (A - LU) = 2.80 \times 10^{-12} $$


<a name='cholesky'></a>
## Cholesky

Suppose we wish to find an LU decomposition for a special type of matrix -- one that is symmetric positive definite. In a $$4 \times 4$$ version, we would see

$$
\begin{bmatrix} A_{11} & A_{12} & A_{13} & A_{14} \\ A_{21} & A_{22} & A_{23} & A_{24} \\ A_{31} & A_{32} & A_{33} & A_{34} \\ A_{41} & A_{42} & A_{43} & A_{44} \end{bmatrix} = \begin{bmatrix} g_{11} & 0 & 0 & 0 \\ g_{21} & g_{22} & 0 & 0 \\ g_{31} & g_{32} & g_{33} & g_{34} \\ g_{41} & g_{42} & g_{43} & g_{44} \end{bmatrix} \begin{bmatrix} g_{11} & g_{21} & g_{31} & g_{41} \\ 0 & g_{22} & g_{32} & g_{42} \\ 0 & 0 & g_{33} & g_{43} \\ 0 & 0 & 0 & g_{44} \end{bmatrix}
$$

This gives us a set of $$4^2=16$$ equations. Note that the equations we obtain from the bottom triangle of the matrix $$A$$, are identical to the equations we obtain from the upper triangle of $$A$$ (since $$A$$ is symmetric). For simplicity, we will work with the equations from the bottom triangle of $$A$$.

When we solve the constraints, we note (with one-indexed matrices), we have the following definitions for each term:

$$
g_{ij} = \frac{a_{ij} - \sum\limits_{k=1}^{j-1} g_{ik}g_{jk} }{g_{jj}}
$$

When $$j=i$$, $$a_{jj} = g_{jj} g_{jj} + \sum\limits_{k=1}^{j-1}g_{jk}^2$$, implying

$$
g_{jj} = \sqrt{a_{jj} - \sum\limits_{k=1}^{j-1}g_{jk}^2}
$$

```python
def Cholesky(A):
    """ 
    Only operate on bottom triangle of the matrix A,
    since A is symmetric (get same constraints
    from upper and lower triangle of A).
    """
    n,_ = A.shape
    G = np.zeros((n,n))

    # populate lower triangular part
    for i in range(n):
        for j in range(n):
            if j > i:
                continue
            k_sum = 0
            if j==i: # diagonal element
                for k in range(j):
                    k_sum += (G[j,k]**2)
                G[j,j] = np.sqrt(A[j,j] - k_sum)
            else:
                for k in range(j):
                    k_sum += G[i,k]*G[j,k]
                G[i,j] = (A[i,j] - k_sum)/G[j,j]

    return G
```

Below we show a simple code example that generates a symmetric positive definite matrix $$A$$ and factorizes it into $$GG^T$$:
```python
def Cholesky_demo():
    n = 100
    # generate a random n x n matrix
    A = np.random.randn(n,n)

    # construct a symmetric matrix using either
    A = 0.5*(A+A.T)
    #A = A.dot(A.T)

    # make symmetric diagonally dominant matrix
    A += n * np.eye(n)
    w,v = np.linalg.eig(A)
    G = Cholesky(A)

    assert(np.allclose(np.linalg.cholesky(A),G))
    assert( (G.dot(G.T) - A ).sum() < 1e-10 )
```




<a name='qr'></a>
## The QR Factorization

The QR Factorization is a matrix factorization especially useful for solving least-squares problems. In this section, I will show you how to compute in Python what you could obtain with a library like Numpy, if you were to call `Q, R = np.linalg.qr(A)`.

Although there are multiple ways to form a QR decomposition, we will use Householder triangularization in this example.  A Householder matrix $$P$$ takes on the form:

$$
P = I_n - \frac{2vv^T}{v^Tv}
$$

```python
def householder(A):
    """     
        Args:
        -   A: A n-d numpy array of shape (n,n) 

        Returns:
        -   P: A n-d numpy array of shape (n,n), representing 
                a Householder matrix
    """
    n, _ = A.shape
    x = A[:,0].reshape(n,1)
    e1 = np.zeros((n,1))
    e1[0] = 1.
    v = np.sign(x[0]) * np.linalg.norm(x) * e1 + x
    # divide by 2-norm squared
    P = np.eye(n) - 2 * v.dot(v.T)/v.T.dot(v)
    return P
```


The QR decomposition consists of iteratively multiplying Householder matrices to upper-triangularize a matrix $$A$$ into $$R$$.
```python
def qr_decomposition(A):
    """ 
    Form the QR decomposition of A by using
    Householder matrices.
    """
    n, _ = A.shape
    P = np.zeros((n-1,n,n))
    R = A
    for i in range(n-1):

        P[i] = np.eye(n)
        # modify lower-right block
        P[i,i:,i:] = householder(R[i:,i:])
        print(f'On iter {i}', np.round(P[i].dot(R), 1))
        R = P[i].dot(R)

    Q_T = np.eye(n)
    for i in range(n-1):
        Q_T = P[i].dot(Q_T)

    Q = Q_T.T
    return Q, R
```

Here is a simple code example, generating a random matrix $$A$$ which we will factorize:
```python
def qr_demo():
    """ """
    n = 4
    A = np.random.randint(-100,100,(n,n))

    Q,R = qr_decomposition(A)
    QR = Q.dot(R)
    print('Sum: ', np.absolute(A - QR).sum())
    assert( (A - QR).sum() < 1e-10 )
```
Suppose we randomly generated a matrix $$A$$, as follows:

$$
A = \begin{bmatrix}
  6  & 6  & -77 & 59 \\
 -13 & 20 & -81 &  1 \\
 -33 & -35 & -65 & -74 \\
  98 & 92 & 42 &  2 \\
\end{bmatrix}
$$

After iteration 0, all matrix entries have changed, but the first column is upper triangular:

$$
P_0 A = \begin{bmatrix}
 -104.4 & -95.3 & -65.6 & -28.5 \\
    0.  &  31.9 & -82.3 &  11.3 \\
    0. &   -4.7 & -68.4 & -47.8 \\
    0.  &   2.1 &  52.1 & -75.7 \\
\end{bmatrix}
$$


After iteration 1, the first two columns are upper triangular and the first row is unmodified:

$$
P_1 P_0 A = \begin{bmatrix}
-104.4 & -95.3 & -65.6 & -28.5 \\
   0 &   -32.3 &  67.9 & -13.3 \\
    0 &    0. &  -79.4 & -46.  \\
    0 &   0.  &  57. &  -76.5 \\
\end{bmatrix}
$$

After iteration 3, the first three columns are upper triangular, making the matrix upper triangular (this is $$R$$):

$$
P_2 P_1 P_0 A = \begin{bmatrix}
-104.4 &  -95.3 &  -65.6 &  -28.5 \\
    0 &   -32.3 &   67.9 &  -13.3 \\
    0 &     0 &    97.8 &   -7.2 \\
    0 &     0 &    0 &   -89 \\
 \end{bmatrix}
$$

We obtain an orthogonal matrix $$Q=(P_2 P_1 P_0)^T$$ and an upper triangular matrix $$R=Q^TA$$:

$$
Q = \begin{bmatrix}
-0.05 & -0.01 & -0.81 & -0.57 \\
  0.12 & -0.98 & -0.06 &  0.10 \\
  0.31 & 0.15 & -0.55 &  0.75 \\
 -0.93 & -0.07 & -0.14  0.30 \\
\end{bmatrix},  R= 
\begin{bmatrix}
-104.4 & -95.3 & -65.6 & -28.5 \\
   0 &  -32.3 &  67.9 & -13.3 \\
    0 &    0  &  97.8 &  -7.2 \\
    0 &    0  &  0 &  -89. 
\end{bmatrix}
$$


Indeed, we have found a factorization $$A=QR$$, up to machine precision:

$$
QR= \begin{bmatrix}
  6. &  6. & -77. & 59. \\
 -13. & 20. & -81. &  1. \\
 -33. & -35. & -65. & -74. \\
  98. & 92. & 42. &  2.
 \end{bmatrix},
   A= \begin{bmatrix}
   6 &  6 & -77 & 59 \\
 -13 & 20 & -81 &  1 \\
 -33 & -35 & -65 & -74 \\
  98 & 92 & 42 &  2 \\
\end{bmatrix}
$$

The summed absolute difference of the matrices is:
```python
np.absolute(A - QR).sum()) = 2.025e-13
```

You can find the Python code [here](/assets/qr_decomposition.py).

We verify that $$Q$$ is orthogonal meaning $$det(Q) = \pm 1$$ and $$Q^TQ=QQ^T=I$$


We showed an example above for a $$n \times n$$ matrix, but this holds for $$m \times n$$ matrices as well. In general, we need $$n$$ Householder matrices (assuming $$m>n$$). Only when $$m=n$$, we need only $$n-1$$ of them. However, in a typical LS problem, we have $$m>n$$.


## References
[1] Lloyd Trefethen and David Bau. Numerical Linear Algebra, SIAM, 2007.














