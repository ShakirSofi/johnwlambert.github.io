---
layout: post
title:  "Solving Least Squares with QR"
permalink: /least-squares/
excerpt: "QR decomposition, Modified Gram Schmidt (MGS)"
mathjax: true
date:   2018-10-02 11:00:00
mathjax: true

---
Table of Contents:
- [Least Squares](#lstqr)
- [QR for Least Squares](#qr-for-lstsqr)
- [MGS for QR](#mgs-for-qr)
- [QR for GMRES](#qr-for-gmres)


<a name='lstqr'></a>
## The Least-Squares Problem

The Least-Squares (LS) problem is one of the central problems in numerical linear algebra. I will describe why. Suppose we have a system of equations $$Ax=b$$, where $$A \in \mathbf{R}^{m \times n}$$, and $$m \geq n$$, meaning $$A$$ is a long and thin matrix and $$b \in \mathbf{R}^{m \times 1}$$. We wish to find $$x$$ such that $$Ax=b$$. In general, we can never expect such equality to hold if $$m>n$$! We can only expect to find a solution $$x$$ such that $$Ax \approx b$$. Formally, the LS problem can be defined as

$$
\mbox{arg }\underset{x}{\mbox{min}} \|Ax-b\|_2 \\ 
$$

## Motivation for QR

The LS problem is simple to define. But how can we find a solution vector $$x$$ in practice, i.e. numerically? Recall our LU decomposition from [our previous tutorial](/direct-methods/).

Gaussian Elimination (G.E.) on non-square matrix -- $$(5 \times 5)(5 \times 3)$$ -- elementary matrix is $$(5 \times 5)$$

$$
\begin{equation}
    \begin{bmatrix} 1 & & & & \\ x & 1& & & \\ x & & 1& & \\ x & & & 1& \\ x & & & &1 \end{bmatrix} \begin{bmatrix} x x x \\ x x x \\ x x x \\ x x x \\ x x x \\ \end{bmatrix} \rightarrow \begin{bmatrix} x & x & x \\ 0 & x & x \\ 0 & 0 & x \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}
\end{equation}
 $$

Even if G.E. goes through on $$A$$ here, i.e. G.E. doesn't break down and we have $$A=LU$$, then we plug in

$$
\begin{aligned}
    &= \underset{x}{\mbox{min}} \|Ax-b\|_2 \\ 
    &= \underset{x}{\mbox{min}} \|LUx-b\|_2         
\end{aligned}
$$

Cannot make the problem much simpler at this point. Need a different approach.

A better way is to rely upon an orthogonal matrix $$Q$$. Assume $$Q \in \mathbf{R}^{m \times m}$$ with $$Q^TQ=I$$. Then $$Q$$ doesn't change the norm of a vector. If you rotate or reflect a vector, then the vector's length won't change. Consider why:

$$
    \|Qy\|_2^2 = (Qy)^T (Qy) = y^TQ^T Qy  =  y^Ty =  \|y\|_2^2
$$

Consider how an orthogonal matrix can be useful in our traditional least squares problem:

$$
\begin{aligned}
    &= \underset{x}{\mbox{min}} \|Ax-b\|_2 \\ 
    &= \underset{x}{\mbox{min}} \|Q^T(Ax-b)\|_2 \\
    &= \underset{x}{\mbox{min}} \| Q^T(QRx-b)\|_2 \\
    &= \underset{x}{\mbox{min}} \|Rx-Q^Tb)\|_2
\end{aligned}
$$

Our goal is to find a $$Q$$ s.t. $$Q^TA = Q^TQR= R$$ is upper triangular.

<a name='qr-for-lstsqr'></a>
### QR Factorization for Solving Least Squares Problems

I'll briefly review the QR decomposition, which **exists for any matrix**.  Given a matrix $$A$$, the goal is to find two matrices $$Q,R$$ such that $$Q$$ is orthogonal and  $$R$$ is upper triangular. If $$m \geq n$$, then

$$
\underbrace{A}_{m \times n} =  \underbrace{Q}_{m \times m} \underbrace{\begin{bmatrix} R \\ 0 \end{bmatrix}}_{m \times n} = \begin{bmatrix} Q_1 & Q_2 \end{bmatrix} \underbrace{\begin{bmatrix} R \\ 0 \end{bmatrix}}_{m \times n}
$$


We call this the *full QR* decomposition. No matter the structure of $$A$$, the matrix $$R$$ will always be square. There is another form, called the *reduced QR* decomposition, of the form:

$$
\underbrace{A}_{m \times n} = \underbrace{Q_1}_{m \times n} \mbox{ } \underbrace{R}_{n \times n}
$$

An important question at this point is *how can we actually compute the QR decomposition* (i.e. numerically)? [We reviewed the Householder method](/direct-methods/) for doing so previously, and will now describe how to use the Gram-Schmidt (GS) to find matrices $$Q,R$$.


<a name='mgs-for-qr'></a>
## Computing QR with Modified Gram Schmidt (MGS)

Computing the reduced QR decomposition of a matrix $$A$$ with the Modified Gram Schmidt (MGS) algorithm requires looking at the matrix $$A$$ with new eyes.

When we view $$A$$ as the product of two matrices, i.e. $$A=Q_1 R$$, then we can also view it as a sum of outer products of the columns of $$Q_1$$ and the rows of $$R$$, i.e.

$$
\begin{aligned}
A &= \begin{bmatrix} | & | & & | \\ q_1 & q_2 & \cdots & q_n \\  | & | & & |  \end{bmatrix}  \begin{bmatrix} - &  \tilde{r}_1^T & - \\ - & \tilde{r}_2^T & - \\ &  \vdots & \\ - & \tilde{r}_n^T & - \end{bmatrix} \\
&=
\begin{bmatrix} | \\ q_1 \\ | \end{bmatrix} \begin{bmatrix} - &  \tilde{r}_1^T  & - \end{bmatrix}
+ 
\begin{bmatrix} | \\ q_2 \\ | \end{bmatrix} \begin{bmatrix} - &  \tilde{r}_2^T  & - \end{bmatrix}
+ \cdots + \begin{bmatrix} | \\ q_n \\ | \end{bmatrix} \begin{bmatrix} - &  \tilde{r}_n^T  & - \end{bmatrix}
\end{aligned}
$$

However, it turns out that each of these outer products has a very special structure, i.e. they each have more columns with all zeros. This is due to the fact that the rows of $$R$$ have a large number of zero elements since the matrix is upper-triangular.
Consider a small example for $$m=5,n=3$$:

$$
\begin{aligned}
\underbrace{A}_{5 \times 3} &= \underbrace{Q_1}_{5 \times 3} \underbrace{R}_{3 \times 3} \\
 A &= \begin{bmatrix} a_1 & a_2 & a_3 \end{bmatrix} \\
 A  &=  \begin{bmatrix} q_1 & q_2 & q_3 \end{bmatrix}  \begin{bmatrix} r_1^T \\ r_2^T \\ r_3^T \end{bmatrix} \\
       &= \begin{bmatrix} q_1 & q_2 & q_3 \end{bmatrix} \begin{bmatrix} \times & \times & \times \\ 0 & \times & \times \\ 0 & 0 & \times \end{bmatrix} \\
      &=  q_1r_1^T + q_2 r_2^T + q_3 r_3^T \\
      A &= \begin{bmatrix}\times & \times & \times \\ \times & \times & \times \\ \times & \times & \times\\ \times & \times & \times \\ \times & \times & \times \end{bmatrix} +
      \begin{bmatrix} 0 & \times & \times \\ 0 & \times & \times  \\ 0 & \times & \times  \\ 0 & \times & \times  \\ 0 & \times & \times  \end{bmatrix} +
      \begin{bmatrix}  0 & 0 & \times \\  0 & 0 & \times \\  0 & 0 & \times \\  0 & 0 & \times \\  0 & 0 & \times \end{bmatrix}  \\
\end{aligned}
$$

where "$$\times$$" denotes a potentially non-zero matrix entry. Consider a very interesting fact: if the equivalence above holds, then by subtracting a full matrix $$q_1r_1^T$$ we are guaranteed to obtain a matrix with at least one zero column. We call the embedded matrix $$A^{(2)}$$:

$$
\begin{aligned}
    \begin{bmatrix} | & \\ 0 & A^{(2)} \\ | & \end{bmatrix} &= A - \begin{bmatrix}\times & \times & \times \\ \times & \times & \times \\ \times & \times & \times\\ \times & \times & \times \\ \times & \times & \times \end{bmatrix} = \begin{bmatrix} 0 & \times & \times \\ 0 & \times & \times  \\ 0 & \times & \times  \\ 0 & \times & \times  \\ 0 & \times & \times  \end{bmatrix} +
      \begin{bmatrix}  0 & 0 & \times \\  0 & 0 & \times \\  0 & 0 & \times \\  0 & 0 & \times \\  0 & 0 & \times \end{bmatrix} \\
\end{aligned}
$$

We can generalize the composition of $$A^{(k)}$$, which gives us the key to computing a column of $$Q$$, which we call $$q_k$$:

$$
\begin{aligned}
      A &= \sum\limits_{i=1}^n q_r r_i^T \\
      A &= \sum\limits_{i=1}^{k-1} q_i r_i^T + \sum\limits_{i=k}^n q_i r_i^T \\
      A - \sum\limits_{i=1}^{k-1} q_i r_i^T &= \sum\limits_{i=k}^n q_i r_i^T \\
      \begin{bmatrix} 0 & A^{(k)} \end{bmatrix} e_k &= \sum\limits_{i=k}^n q_i r_i^T e_k \\
      &= q_k r_k^T e_k + q_{k+1} r_{k+1}^T e_k + \cdots + qq_n r_n^T e_k \\
      &= q_k r_{kk} 
\end{aligned}
$$

We multiply with $$e_k$$ above simply because we wish to compare the $$k$$'th column of both sides. For ease of notation, we will call the first column of $$A^{(k)}$$ to be $$z$$:

$$
\begin{aligned}
      \underbrace{ A^{(k)} }_{m \times (n-k+1)} &= \begin{bmatrix} z & B \end{bmatrix} \\
     z &=  q_k r_{kk} \\
     q_k &= z / r_{kk} \\
     r_{kk} &= \|z\|_2      
\end{aligned}
$$

where $$B$$ has $$(n-k)$$ columns. You will find $$(k-1)$$ zero columns in $$A - \sum\limits_{i=1}^{k-1} q_i r_i^T $$. 

A second key observation allows us to compute the entire $$k$$'th row $$\tilde{r}^T$$ of $$R$$ just by knowing $$q$$. Consider what would happen if we left multiply with $$ q_k^T$$: since the columns of $$Q$$ are all orthogonal to each other, their dot product will always equal zero, unless $$i=k$$, in which case $$q_k^T q_k = 1$$:
   
\begin{equation}
       q_k^T \begin{bmatrix} 0 & A^{(k)} \end{bmatrix} = q_k^T \Bigg( \sum\limits_{i=k}^n q_i r_i^T \Bigg) = r_k^T
\end{equation}

Since a row of $$R$$ is upper triangular, all elements $$R_{ij}$$ where $$j < i$$ will equal zero:
   
\begin{equation}
       q_k^T \begin{bmatrix} 0 & z & B \end{bmatrix} = \begin{bmatrix} 0 & \cdots & 0 & r_{kk} & r_{k,k+1} \cdots & r_{kn} \end{bmatrix}
\end{equation}

which is the $$k$$'th row of $$R$$. When $$k=1$$:
   
$$
\begin{aligned}
A &= \sum\limits_{i=1}^n q_i r_i^T \\
&= q_1r_1^T + \cdots + q_n r_n^T \\
Ae_1 &= \Big( q_1 r_1^T + \cdots + q_n r_n^T \Big) e_1 & \mbox{take the first colum of both sides} \\
Ae_1 &= q_1 r_{11} \\
a_1 &= Ae_1 = q_1 r_{11} \\
r_{11} &= \|a_1 \|_2 \\
q_1 &= a_1 / r_{11} \\
q_1^T A &= q_1^T (q_1 r_1^T + \cdots + q_n r_n^T) = r_1^T \\
A - q_1 r_1^T &= \sum\limits_{i=2}^n q_i r_i^T
\end{aligned}
$$

We can use induction to prove the correctness of the algorithm. We know how to deal with this when $$k=1$$

   
   
\begin{equation}
       a_1 = Ae_1 = \sum\limits_{i=1}^n q_i r_i^T e_1 = q_1 r_{11}
\end{equation}
   
\begin{equation}
   q_1^T A = q_1^T ( \sum\limits_{i=1}^n q_i r_i^T) = r_1^T
\end{equation}
   
\begin{equation}
 \begin{bmatrix} 0 & A^{(2)} \end{bmatrix} = A - q_1 r_1^T = \sum\limits_{i=2}^n q_i r_i^T
\end{equation}


## Why is this identical to Modified Gram-Schmidt?


The Gram-Schmidt (GS) method has two forms: classical and modifed
It is for  $$\underbrace{A}_{m \times n}=\underbrace{Q_1}_{m \times n} \underbrace{R}_{n \times n}$$ when $$rank(A)=n$$ (full rank case).

$$
A = \begin{bmatrix} a_1 & \cdots & a_n \end{bmatrix} = Q_1R = \begin{bmatrix} q_1 & \cdots & q_n \end{bmatrix} \begin{bmatrix} r_{11} & r_{12} & \cdots & r_{1n} \\ & r_{22} & \cdots & r_{2n} \\ & & \ddots & \vdots \\ & & & r_{nn} \end{bmatrix}
$$


$$
\begin{aligned}
    a_1 = r_{11} q_1 \\
    q_1 = \frac{a_1}{\|a_1\|_2}
\end{aligned}
$$


GE for LU
\begin{equation}
    M_{n-1} \cdots M_1 \underbrace{A}_{n \times n} = U
\end{equation}

\item Householder for QR:
\begin{equation}
    H_n \cdots H_1 \underbrace{A}_{m \times n, m > n} = \begin{bmatrix} R  \\ 0 \end{bmatrix}
\end{equation}

\item Givens for QR:
\begin{equation}
G_{pq} \cdots G_{12} \underbrace{A}_{m \times n, m>n} = \begin{bmatrix} R \\ 0 \end{bmatrix}
\end{equation}

Gram Schmidt: Have to compute $$Q_1,R$$ at the same time. Cannot skip computing $$Q$$. Assuming $$rank(A)=n$$
    
\begin{equation}
        \underbrace{A}_{m \times n} = \underbrace{Q_1}_{m \times n} \underbrace{R}_{n \times n}
\end{equation}

Classical Gram Schmidt: compute column by column

\begin{equation}
        \begin{bmatrix} a_1 & \cdots & a_n \end{bmatrix} = \begin{bmatrix} q_1 & \cdots & q_n \end{bmatrix} \begin{bmatrix} r_{11} & r_{12} & \cdots & r_{1n} \\ & & & \\ & & & r_{nn} \end{bmatrix}
\end{equation}

If you skip computing columns of $Q$, you cannot continue.

Classical GS (CGS) can suffere from cancellation error. Nearly equal numbers (of same sign) involved in subtraction. If two vectors point in almost the same direction. $p_2$ could have very low precision
   
Modifed Gram Schmidt is just order re-arrangement! Gram Schmidt is only for full-rank matrices.


<a name='qr-for-gmres'></a>
## QR Application: The GMRES Algorithm

The Generalized Minimum Residual (GMRES) algorithm, a classical iterative method for solving very large, sparse linear systems of equations relies heavily upon the QR decomposition. GMRES [1] was proposed by Usef Saad and Schultz in 1986, and has been cited $$>10,000$$ times.

The least squares optimization problem of interest in GMRES is

$$
\underset{y}{\mbox{min}} \| \beta \xi_1 - H_{k+1,k} y \|
$$

We choose $$y$$ such that the sum of squares is minimized. The following code computes the QR decomposition to solve the least squares problem.


```python
def arnoldi_single_iter(A,Q,k):
	""" 
		Args:
		-	A
		-	Q
		-	k

		Returns:
		-	h
		-	q
	"""
	q = A.dot(Q[:,k])
	h = np.zeros(k+2)
	for i in range(k+1):
		h[i] = q.T.dot(Q[:,i])
		q -= h[i]*Q[:,i]
	h[k+1] = np.linalg.norm(q)
	q /= h[k+1]
	return h,q
```


```python
def gmres(A,b,x, max_iters):
	""" 
	Generalized Minimal Residual Algorithm.

		Args:
		-	A: must be square and nonsingular
		-	b
		-	x: initial guess for x

	"""
	EPSILON = 1e-10
	n,_ = A.shape
	assert(A.shape[0]==A.shape[1])

	r = b - A.dot(x)
	q = r / np.linalg.norm(r)
	Q = np.zeros((n,max_iters))
	Q[:,0] = q.squeeze()
	beta = np.linalg.norm(r)
	xi = np.zeros((n,1))
	xi[0] = 1 # e_1 standard basis vector, xi will be updated
	H = np.zeros((n+1,n))

	F = np.zeros((max_iters,n,n))
	for i in range(max_iters):
		F[i] = np.eye(n)

	for k in range(max_iters-1):
		H[:k+2,k], Q[:,k+1] = arnoldi_single_iter(A,Q,k)

		# don't need to do this for 0,...,k since completed previously!
		c,s = givens_coeffs(H[k,k], H[k+1,k])
		# kth rotation matrix
		F[k, k,k] = c
		F[k, k,k+1] = s
		F[k, k+1,k] = -s
		F[k, k+1,k+1] = c

		# apply the rotation to both of these
		H[:k+2,k] = F[k,:k+2,:k+2].dot(H[:k+2,k])
		xi = F[k].dot(xi)

		if beta * np.linalg.norm(xi[k+1]) < EPSILON:
			# ????
			break

	# when terminated, solve the least squares problem
	# y must be (k,1)
	y, _, _, _ = np.linalg.lstsq(H[:k+1,:k+1],xi[:k+1])
	# Q_k will have dimensions (n,k)
	x_k = x + Q[:,:k+1].dot(y)
	return x_k
```


```python
def givens_coeffs(a,b):
	""" """
	c = a / np.sqrt(a**2 + b**2)
	s = b / np.sqrt(a**2 + b**2)
	return c,s
```




	
```python
def arnoldi(A,b,k):
	""" 
	Computes a basis of the (k+1)-Krylov subspace of A: the space
	spanned by {b, Ab, ..., A^k b}.

		Args:
		-	A: Numpy array of shape (n,n)
		-	b: 
		-	k: dimension of Krylov subspace

		Returns:
		-	Q: Orthonormal basis for Krylov subspace
		-	H: Upper Hessenberg matrix
	"""
	n = A.shape[0]

	H = np.zeros((k,k))
	Q = np.zeros((n,k))

	# Normalize the input vector
	# Use it as the first Krylov vector
	Q[:,0] = b / np.linalg.norm(b)

	for j in range(k-1):
		Q[:,j+1] = A.dot(Q[:,j])
		for i in range(j):
			H[i,j] = Q[:,j+1].dot(Q[:,i])
			Q[:,j+1] = Q[:,j+1] - H[i,j] * Q[:,i]

		H[j+1,j] = np.linalg.norm(Q[:,j+1])
		Q[:,j+1] /= H[j+1,j]
	pdb.set_trace()
	return Q,H
```

## References

[1.] Y Saad, MH Schultz. *GMRES: A generalized minimal residual algorithm for solving nonsymmetric linear systems}*.  SIAM Journal on scientific and statistical computing 7 (3), 856-869. [PDF](https://epubs.siam.org/doi/pdf/10.1137/0907058). 

