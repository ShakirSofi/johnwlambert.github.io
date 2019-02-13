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
- [Back Substitution](#sfmpipeline)
- [LU Factorization](#costfunctions)
- [Cholesky Factorization](#bundleadjustment)

<a name='sfmpipeline'></a>

## Back Substitution for Triangular Matrices

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



## Cholesky

























