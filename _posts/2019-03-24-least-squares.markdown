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



## Computing QR with Givens Rotations







<a name='mgs-for-qr'></a>
## Computing QR with Modified Gram Schmidt (MGS)

Computing the reduced QR decomposition of a matrix $$\underbrace{A}_{m \times n}=\underbrace{Q_1}_{m \times n} \underbrace{R}_{n \times n}$$ with the Modified Gram Schmidt (MGS) algorithm requires looking at the matrix $$A$$ with new eyes.

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


## Why is this identical to Modified Gram-Schmidt (MGS)?

We stated that the process above is the "MGS method for QR factorization". It might not be clear why the process is equivalent to MGS.

First, let's review the Gram-Schmidt (GS) method, which has two forms: classical and modifed. Gram-Schmidt is only a viable way to obtain a QR factorization when A is full-rank, i.e. when $$rank(A)=n$$. This is because at some point in the algorithm we exploit linear independence, which, when violated, means we divide by a zero.

$$
A = \begin{bmatrix} a_1 & \cdots & a_n \end{bmatrix} = Q_1R = \begin{bmatrix} q_1 & \cdots & q_n \end{bmatrix} \begin{bmatrix} r_{11} & r_{12} & \cdots & r_{1n} \\ & r_{22} & \cdots & r_{2n} \\ & & \ddots & \vdots \\ & & & r_{nn} \end{bmatrix}
$$


$$
\begin{aligned}
    a_1 = r_{11} q_1 \\
    q_1 = \frac{a_1}{\|a_1\|_2}
\end{aligned}
$$

## How does MGS compare with the Givens and Householder methods for QR?

MGS is certainly not the only method we've seen so far for finding a QR factorization. We discussed the Householder method (earlier)[/direct-methods/#qr], which finds a sequence of orthogonal matrices $$H_n \cdots H_1$$ such that

$$
    H_n \cdots H_1 \underbrace{A}_{m \times n, m > n} = \begin{bmatrix} R  \\ 0 \end{bmatrix}
$$

We have also seen the Givens rotations, which find another sequence of orthogonal matrices $$G_{pq} \cdots G_{12}$$ such that

$$
G_{pq} \cdots G_{12} \underbrace{A}_{m \times n, m>n} = \begin{bmatrix} R \\ 0 \end{bmatrix}
$$

In these methods, it was possible to skip the computation of $$Q$$ explicitly. However, in Gram-Schmidt this is not the case: we must compute $$Q_1,R$$ at the same time and we cannot skip computing $$Q$$. In fact, if you skip computing columns of $$Q$$, you cannot continue.
    
$$
      \underbrace{A}_{m \times n} = \underbrace{Q_1}_{m \times n} \underbrace{R}_{n \times n}
$$

Classical Gram Schmidt: compute column by column

$$
 \begin{bmatrix} a_1 & \cdots & a_n \end{bmatrix} = \begin{bmatrix} q_1 & \cdots & q_n \end{bmatrix} \begin{bmatrix} r_{11} & r_{12} & \cdots & r_{1n} \\ & & & \\ & & & r_{nn} \end{bmatrix}
$$


Classical GS (CGS) can suffer from cancellation error. Nearly equal numbers (of same sign) involved in subtraction. If two vectors point in almost the same direction. $p_2$ could have very low precision
   
Modifed Gram Schmidt is just order re-arrangement! 


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

## Rank-Deficient Least-Squares Problems
When we used the QR decomposition of a matrix $$A$$ to solve a least-squares problem, we operated under the assumption that $$A$$ was full-rank. This assumption can fall flat. In that case we revert to rank-revealing decompositions. Suitable choices are either the (1) SVD or its cheaper approximation, (2) QR with column-pivoting.


You might ask, why is the rank-deficient case problematic?


If matrix $A$ is rank-deficient, then it is no longer the case that space spanned by columns of $Q$ is the same space spanned by columns of $A$, i.e. does not hold that
    
\begin{equation}
        \mbox{span} \{ a_1, a_2, \cdots, a_k \} =  \mbox{span} \{ q_1, q_2, \cdots, q_k \}
\end{equation}

### SVD for Least-Squares

As stated above, we should use the SVD when we don't know the rank of a matrix, or when the matrix is known to be rank-deficient.

$$
A &= U \Sigma V^T
&= \begin{bmatrix} U_1 & U_2 \end{bmatrix} \begin{bmatrix} \Sigma_1 & 0 \\ 0 & 0 \end{bmatrix} \begin{bmatrix} V_1 & V_2 \end{bmatrix}^T
$$

$$
\mbox{arg} \underset{x}{\mbox{ arg }} \| Ax -b \|_2 \\

$$

We can write the SVD as
    
$$
\begin{aligned}
    A &= U \Sigma V^T \\
      &= \begin{bmatrix} \underbrace{U_1}_{r} & \underbrace{U_2}_{m-r} \end{bmatrix} \begin{bmatrix} \underbrace{\Sigma_1}_{r \times r} & \underbrace{0}_{n-r} \\ 0 & 0 \end{bmatrix} \begin{bmatrix} \underbrace{V_1}_{r} & \underbrace{V_2}_{n-r} \end{bmatrix}^T
\end{aligned}
$$

\item Note that the range space of $A$ is completely spanned by $U_1$! Because everything in $U_2$ has rank 0 because of zero singular vectors
\item Range space of $A^T$ is completely spanned by $V_1$!
\item The null space of $A$ is spanned by $V_2$! We recall that nullspace is defined as $Null(A) = \{ x \mid Ax = 0 \}$

$$
\begin{aligned}
A V_2 &= \begin{bmatrix} U_1 & U_2 \end{bmatrix} \begin{bmatrix} \Sigma_1 & 0 \\ 0 & 0 \end{bmatrix} \begin{bmatrix} V_1^T \\ V_2^T \end{bmatrix} V_2 \\
A V_2 &= \begin{bmatrix} U_1 & U_2 \end{bmatrix} \begin{bmatrix} \Sigma_1 & 0 \\ 0 & 0 \end{bmatrix} \begin{bmatrix} 0 \\ I \end{bmatrix} \\
A V_2 &= \begin{bmatrix} U_1 & U_2 \end{bmatrix}  0 \\
AV_2 &= 0
\end{aligned}
$$

because $V_2 V_1 = 0$ (the zero matrix since must be orthogonal columns)

$$
\begin{aligned}
    A^T &= V \Sigma U^T \\
     A^T &= \begin{bmatrix} \underbrace{V_1}_{r} & \underbrace{V_2}_{m-r} \end{bmatrix} \begin{bmatrix} \underbrace{\Sigma_1}_{r \times r} & \underbrace{0}_{n-r} \\ 0 & 0 \end{bmatrix} \begin{bmatrix} \underbrace{U_1}_{r} & \underbrace{U_2}_{n-r} \end{bmatrix}^T
\end{aligned}
$$

Null space of $A^T$ is spanned by $U_2$! Then in Least Squares, we have
    
$$
\begin{aligned}
        \|Ax-b\|_2 &= \|U \Sigma V^T x - b \|_2 \\
        &= \|U^T(U \Sigma V^T x - b) \|_2 \\
       &= \| \Sigma V^T x - U^Tb \|_2 \\
        &= \| \begin{bmatrix} \Sigma_1 & 0 \\ 0 & 0 \end{bmatrix} V^T x - U^Tb \|_2 \\
        &= \| \begin{bmatrix} \Sigma_1 & 0 \\ 0 & 0 \end{bmatrix} \begin{bmatrix} y \\ z \end{bmatrix} - \begin{bmatrix} c \\ d \end{bmatrix} \|_2 \\
        &= \| \begin{bmatrix} \Sigma_1 y & 0 \\ 0 & 0 \end{bmatrix}  - \begin{bmatrix} c \\ d \end{bmatrix} \|_2 \\
\end{aligned}
$$

and $$z$$ will not affect the solution. We search for $$\underbrace{\Sigma_1}_{r \times r} \underbrace{y}_{r \times 1} = \underbrace{c}_{r \times 1}$$.
    
$$
V^Tx = \begin{bmatrix} V_1^T \\ V_2^T \end{bmatrix} x = \begin{bmatrix} y \\ z \end{bmatrix}
$$
    
$$
 U^Tb = \begin{bmatrix} U_1^Tb \\ U_2^Tb \end{bmatrix} = \begin{bmatrix} c \\ d \end{bmatrix}
$$
    where $c,y $ have shape $r$, and $z,d$ have shape $n-r$.

The mininimum norm solution is 
    
$$
V \begin{bmatrix} \Sigma_1^{-1}c \\ 0 \end{bmatrix}
$$

There are infinitely many solutions. If you put a non-zero element in the second part (instead of $$0$$), then it no longer has the smallest norm
   
When you split up a matrix $Q$ along the rows, then you should keep in mind that the columns will still be orthogonal to each other, but they won't have unit length norm any more (because not working with the full row)
   
But we wanted to find a solution for $$x$$, not $$y$$! Thus, we do
   
$$
\begin{aligned}
       x_{LS} = V \begin{bmatrix} y \\ z \end{bmatrix} \\
       x_{LS} = \begin{bmatrix} V_1 & V_2 \end{bmatrix} \begin{bmatrix} y \\ z \end{bmatrix} \\
       &= V_1 y + V_2 z
\end{aligned}
$$

where $$z$$ can be anything -- it is a free variable!

We recall that if $$A$$ has dimension $$(m \times n)$$, with $$m > n$$, and $$rank(a)< n $$, then $\exists$$ infinitely many solutions

Meaning that $$x^{\star} + y$ is a solution when $y \in null(A)$ because $$A(x^{\star} + y) = Ax^{\star} + Ay = Ax^{\star}$$



### QR with Column-Pivoting

Computing the SVD of a matrix is an expensive operation. A cheaper alternative is *QR with column-pivoting*. Recall Guassian Elimination (G.E.) with complete pivoting (i.e. pivoting on both the rows and columns), which computes a decomposition:
$$ P A \Pi = L U $$. G.E. with only column pivoting would be defined as $$  A \Pi = LU $$

Consider applying the pivoting idea to the full, non-reduced QR decomposition, i.e. 

$$
\begin{aligned}
A &= QR \\
A \Pi &= QR \Pi \\
\underbrace{A}_{m \times n} &=  \underbrace{Q}_{m \times m} \underbrace{\begin{bmatrix} R \\ 0 \end{bmatrix}}_{m \times n} \Pi \\
\underbrace{A}_{m \times n} \underbrace{\Pi}_{n \times n} &= \underbrace{Q}_{m \times m} \begin{bmatrix} \underbrace{R_{11}}_{(r \times r)} & \underbrace{R_{12}}_{(n-r) \times r} \\ 0 & 0 \end{bmatrix}
\end{aligned}
$$

An immediate consequence of swapping the columns of an upper triangular matrix $$R$$ is that the result has no upper-triangular guarantee. In our case, the we call the result $$\begin{bmatrix} R_{11} & R_{12} \\ 0 & 0 \end{bmatrix}$$, where $$r = rank(A)$$, and $$rank(R_{11}) = r$$.

Note that in the decomposition above, $$Q$$ and $$\Pi$$ are both orthogonal matrices. Thus, this decomposition has some similarities with the SVD decomposition $$A=U \Sigma V^T$$, which is composed of two orthogonal matrices $$U,V$$. Multiplying by $$Q^T = Q^{-1}$$ and $$V^T = V^{-1}$$, we find:

$$
U^T A V = \begin{bmatrix} \Sigma_1 & 0 \\ 0 & 0 \end{bmatrix}
$$

In our *QR with column-pivoting* decomposition, we also see two orthogonal matrices on the left, surrounding $$A$$:

$$
\begin{aligned}
Q^T A \Pi &= \begin{bmatrix} R_{11} & R_{12} \\ 0 & 0 \end{bmatrix} \\
Q^T A \Pi \Pi^{T} &= \begin{bmatrix} R_{11} & R_{12} \\ 0 & 0 \end{bmatrix} \Pi^T & \mbox{since } \Pi \mbox{ is orthogonal, } P^{-1} = P^T \\
A &= Q \begin{bmatrix} R_{11} & R_{12} \\ 0 & 0 \end{bmatrix} \Pi^T & \mbox{since } Q \mbox{ is orthogonal.}
\end{aligned}
$$

Note that $$\Pi$$ is a very restrictive orthogonal transformation. It turns out we can also use this decomposition to solve least squares problems, just as we did with the SVD.
  
    
$$
\begin{aligned}
& \mbox{arg }\underset{x}{\mbox{min}}   \| Ax -b \|_2 \\
&= \mbox{arg }\underset{x}{\mbox{min}}  \| \Bigg( Q \begin{bmatrix} R_{11} & R_{12} \\ 0 & 0 \end{bmatrix} \Pi^T \Bigg) x - b \|_2 \\
&= \mbox{arg }\underset{x}{\mbox{min}}  \| Q^T \Bigg( Q \begin{bmatrix} R_{11} & R_{12} \\ 0 & 0 \end{bmatrix} \Pi^T  x - b \Bigg)\|_2 \\
&= \mbox{arg }\underset{x}{\mbox{min}} \| \begin{bmatrix}
        R_{11} & R_{12} \\ 0 & 0 
        \end{bmatrix}\Pi^Tx - Q^Tb\| \\
\end{aligned}
$$

At this point we'll define new variables for ease of notation. Let $$ Q^Tb = \begin{bmatrix} c \\ d \end{bmatrix}$$ and let  $$\Pi^T x = \begin{bmatrix} y \\ z \end{bmatrix}$$.

Substituting in these new variable definitions, we find

$$
\begin{aligned}
&= \mbox{arg }\underset{x}{\mbox{min}} \| \begin{bmatrix}
        R_{11} & R_{12} \\ 0 & 0 
        \end{bmatrix}\Pi^Tx - Q^Tb\| \\
        &= \mbox{arg }\underset{x}{\mbox{min}} \| \begin{bmatrix}
        R_{11} & R_{12} \\ 0 & 0 
        \end{bmatrix} \begin{bmatrix} y \\ z \end{bmatrix} - \begin{bmatrix} c \\ d \end{bmatrix} \| \\
&= \mbox{arg }\underset{x}{\mbox{min}} \| \begin{bmatrix}
        R_{11}y + R_{12}z - c \\ -d
        \end{bmatrix} \|_2 \\
\end{aligned}
$$
    
We can always solve this equation for $$y$$:
    
\begin{equation}
    R_{11}y  = c - R_{12}z
\end{equation}

which is just a vector with $$r$$ components. We must prove that $$y,z$$ exist such that
    
\begin{equation}
        R_{11}y + R_{12}z - c = 0
\end{equation}

The answer is this is possible. We can make

$$
 \|Ax-b\|_2^2 = \underbrace{ \|R_{11}y + R_{12}z - c \|_2^2 }_{\text{can make }0} + \|d\|_2^2
$$

Choose any $$z$$ and 

$$
y_{ls} = R_{11}^{-1}(c-R_{12}z)
$$

When $$z=0$$, then $$y_{ls}= R_{11}^{-1}c$$. Thus we have a least-squares solution for $$y$$. However, our goal is to find a least-squares solution for $$x$$. We can connect $$x$$ to $$y$$ through the following expressions:

$$
\begin{aligned}
\Pi^T x &= \begin{bmatrix} y \\ z \end{bmatrix} \\
 x &= \Pi \begin{bmatrix} y \\ z
\end{bmatrix} \\
 x &= \Pi \begin{bmatrix} y_{ls} \\ z
\end{bmatrix} \\
      x_{ls} &= \Pi \begin{bmatrix}
         R_{11}^{-1}(c-R_{12}z) \\ z
         \end{bmatrix}
\end{aligned}
$$

The convention is to choose the *minimum norm* solution, which means that $$\|x\|$$ is smallest. The norm of $$x$$ can be computed as follows:

$$
\begin{aligned}
\|x\|_2 = \| \Pi \begin{bmatrix}
         R_{11}^{-1}(c-R_{12}z) \\ z
         \end{bmatrix} \|_2 \\
          \|x\|_2 = \|  \begin{bmatrix}
         R_{11}^{-1}(c-R_{12}z) \\ z
         \end{bmatrix} \|_2
\end{aligned}
$$


## Using Householder Matrices to Compute QR with Column Pivoting



Suppose we started with

$$
\begin{bmatrix}
        x & x & x &x \\
         x & x & x &x \\
          x & x & x &x \\
           x & x & x & x \\
            x & x & x &x \\
\end{bmatrix}
$$

and with permutation annihilation:

$$
Q_2 Q_1 A \Pi_1 \Pi_2 = \begin{bmatrix}
        x & x & x &x \\
         0 & x & x &x \\
          0 & 0 & x &x \\
           0 & 0 & x &x \\
            0 & 0 & x &x \\
        \end{bmatrix}
$$

Already obvious it has rank two. If the matrix was a a total of rank 2, then we know that we really have
    
$$
\begin{bmatrix}
        x & x & x &x \\
         0 & x & x &x \\
          0 & 0 & 0 &0 \\
           0 & 0 & 0 &0 \\
            0 & 0 & 0 &0 \\
\end{bmatrix}
$$

otherwise we would have rank 3! What should be the permutation criteria? SVD rotates all of the mass from left and right so that it is collapsed onto the diagonal:
    
$$
\begin{aligned}
        a = U \Sigma V^T \\
        U^T A V = \Sigma \\
        \cdots U_2^T U_1^T A V_1 V_2 \cdots = \mbox{diagonal matrix}
\end{aligned}
$$

Suppose you do QR without pivoting, then first step of Householder, all of the norm of the entire first column is left in the $$A_{11}$$ entry (top left entry)

$$\Pi_1$$ moves the column with the largest $$\ell_2$$ norm to the 1st column. If we do this, then no matter which column had the largest norm, then the resulting $$A_{11}$$ element will be as large as possible!

We want to move the mass to the left upper corner, so that if the rank is rank-deficient, this will be revealed in the bottom-left tailing side.


## Numerical Instability of the Normal Equations

A popular choice for solving least-squares problems is the use of the *Normal Equations*. Despite its ease of implementation, this method is not recommended due to its numerical instability. The method involves left multiplication with $$A^T$$, forming a square matrix that can (hopefully) be inverted:

$$ 
\begin{aligned}
Ax &= b \\
A^T A x &= A^Tb & \mbox{left multiply with } A^T \\
x &= (A^TA)^{-1} A^Tb & \mbox{invert } (A^TA) \mbox{ and left multiply with } (A^TA)^{-1}\\
\end{aligned}
$$

By forming the product $$A^TA$$, we square the condition number of the problem matrix. Thus, using the QR decomposition yields a better least-squares estimate than the *Normal Equations* in terms of solution quality.


## References

[1.] Y Saad, MH Schultz. *GMRES: A generalized minimal residual algorithm for solving nonsymmetric linear systems}*.  SIAM Journal on scientific and statistical computing 7 (3), 856-869. [PDF](https://epubs.siam.org/doi/pdf/10.1137/0907058). 

