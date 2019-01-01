---
layout: post
title:  "Conjugate Gradients"
permalink: /conjugate-gradients/
excerpt: "large systems of equations, Krylov subspaces, Cayley-Hamilton Theorem"
mathjax: true
date:   2018-12-27 11:00:00
mathjax: true

---
Table of Contents:
- [Conjugate Gradients](#conjugate-gradients)
- [Krylov Subspaces](#krylov-subspaces)
- [Cayley-Hamilton Theorem](#cayley-hamilton-thm)
- [The CG Algorithm](#cg-algo)
- [Preconditioning](#precondition)

<a name='conjugate-gradients'></a>

## Conjugate Gradients (CG)

Conjugate Gradients (CG) is a well-studied method introduced in 1952 by Hestenes and Stiefel [4] for solving a system of $$n$$ equations, $$Ax=b$$, where $$A$$ is positive definite. Solving such a system becomes quite challenging when $$n$$ is large. CG continues to be used today, especially in state-of-the-art reinforcement learning algorithms like [TRPO](/policy-gradients/trunc-natural-grad).

The CG method is interesting because we never give a set of numbers for $$A$$. In fact, we never form or even store the matrix $$A$$. This could be desirable when $$A$$ is huge. Instead, CG is simply a method for calculating $$A$$ times a vector [1] that relies upon a deep result, the *Cayley-Hamilton Theorem*.

A common example where CG proves useful is minimization of the following quadratic form, where $$A \in \mathbf{R}^{n \times n}$$:

$$
f(x) = \frac{1}{2} x^TAx - b^Tx
$$

$$f(x)$$ is a convex function when $$A$$ is positive definite. The gradient of $$f$$ is:

$$
\nabla f(x) = Ax - b
$$

Setting the gradient equal to $$0$$ allows us to find the solution to this system of equations, $$x^{\star} = A^{-1}b$$. CG is an appropriate method to do so when inverting $$A$$ directly is not feasible.

<a name='krylov-subspaces'></a>

## The Krylov Subspace

CG relies upon an idea named the *Krylov subspace*. The Krylov subspace is defined as the span of the vectors generated from successively higher powers of $$A$$:

$$
\mathcal{K}_k = \mbox{span} \{b, Ab, \dots, A^{k-1}b \}
$$

The Krylov sequence is a sequence of solutions to our convex objective $$f(x)$$. Each successive solution has to come from a subspace $$\mathcal{K}_k$$ of progressively higher power $$k$$. Formally, the Krylov sequence $$x^{(1)},x^{(2)},\dots$$ is defined as:

$$
x^{(k)} = \underset{x \in \mathcal{K}_k}{\mbox{argmin }} f(x)
$$

The CG algorithm generates the Krylov sequence.

**Properties of the Krylov Sequence**
It should be clear that $$x^{(k)}=p_k(A)b$$, where $$p_k$$ is a polynomial with degree $$p_k < k$$, because $$x^{(k)} \in \mathcal{K}_k$$. Surprisingly enough, the Krylov sequence is a two-term recurrence:

$$
x^{(k+1)} = x^{(k)} + \alpha_k r^{(k)} + \beta_k (x^{(k)} - x^{(k+1)})
$$

for some values $$\alpha_k, \beta_k$$.This means the current iterate is a linear combination of the previous two iterates.  This is the basis of the CG algorithm.

<a name='cayley-hamilton-thm'></a>

## Cayley-Hamilton Theorem

The reason the Krylov subspace is helpful is because $$x^{\star} = A^{-1}b \in \mathcal{K}_n$$. I will show why. The Cayley-Hamilton Theorem states that if $$A \in \mathbf{R}^{n \times n}$$:

$$
A^n + \alpha_1 A^{n-1} + \cdots + \alpha_n I = 0
$$

We can solve for $$I$$ by rearranging terms:

$$
 \alpha_n I = - A^n + - \alpha_1 A^{n-1} + \cdots + \alpha_{n-1} A^{1}
$$

We now divide by $$\alpha_n$$:

$$
I = - \frac{1}{\alpha_n} A^n + -  \frac{\alpha_1}{\alpha_n} A^{n-1} + \cdots +  \frac{\alpha_{n-1}}{\alpha_n} A^{1}
$$

We now left-multiply all terms by $$A^{-1}$$ and simplify:

$$
\begin{aligned}
A^{-1}I & = - \frac{1}{\alpha_n} A^{-1} A^n + -  \frac{\alpha_1}{\alpha_n} A^{-1} A^{n-1} + \cdots +  \frac{\alpha_{n-1}}{\alpha_n} A^{-1} A^{1} \\
A^{-1} & = - \frac{1}{\alpha_n} A^{n-1} + -  \frac{\alpha_1}{\alpha_n}  A^{n-2} + \cdots +  \frac{\alpha_{n-1}}{\alpha_n} I \\
\end{aligned}
$$

We can now see that $$x^{\star}$$ is a linear combination of the vectors that span the Krylov subspace. Thus, $$x^{\star} = A^{-1}b \in \mathcal{K}_n$$.

<a name='cg-algo'></a>

## The CG Algorithm

We will maintain the square of the residual $$r$$ at each step, which we call $$r_k$$. If the square root of your residual is small enough, $$\sqrt{\rho_{k-1}}$$, then you can quit. Your search direction is $$p$$: the combination of your current r esidual and the previous search direction [2,3].

+ \\(x:=0\\),  \\(r:=b\\),  \\( \rho_0 := \| r \|^2 \\)
+ for $$k=1,\dots,N_{max}$$
	- quit if $$\sqrt{\rho_{k-1}} \leq \epsilon \|b\|$$
	- if \\(k=1\\)
		- \\( p:= r \\)
	- else
		- \\( p:= r + \frac{\rho_{k-1}}{\rho_{k-2}} \\)
	- \\( w:=Ap \\)
	- \\( \alpha := \frac{ \rho_{k-1} }{ p^Tw } \\)
	- \\( x := x + \alpha p \\)
	- \\( r := r - \alpha w \\)
	- \\( \rho_k := \|r\|^2 \\)

Along the way, we've created the Krylov sequence $$x$$. Operations like $$x + \alpha p$$ or $$r - \alpha w$$ are $$\mathcal{O}(n)$$ (BLAS level-1).

<a name='precondition'></a>

## Preconditioning

It turns out that CG will often just fail. The trick in CG is to change coordinates first (precondition) and then run CG on the system in the changed coordinates. This is because of round-off errors that accumulate, leading to unstability and divergence. For example, we may want to make the spectrum of $$A$$ clustered.

A generic preconditioner is a diagonal matrix, e.g.

$$
M = \mbox{diag}(\frac{1}{A_{11}}, \dots, \frac{1}{A_{nn}})
$$


## References

[1] Stephen Boyd. *Conjugate Gradient Method*. EE364b Lectures Slides, Stanford University. [https://web.stanford.edu/class/ee364b/lectures/conj_grad_slides.pdf](https://web.stanford.edu/class/ee364b/lectures/conj_grad_slides.pdf). [Video](https://www.youtube.com/watch?v=E4gl91l0l40).

[2] C.T. Kelley. *Iterative Methods for Optimization*. SIAM, 2000.

[3] C. Kelley. *Iterative Methods for Linear and Nonlinear Equations*. Frontiers in Applied Mathematics SIAM, (1995). [https://archive.siam.org/books/textbooks/fr16_book.pdf](https://archive.siam.org/books/textbooks/fr16_book.pdf).

[4] Magnus Hestenes and Eduard Stiefel. *Method of Conjugate Gradients for Solving Linear Systems*. Journal of Research of the National Bureau of Standards, Vol. 49, No. 6, December 1952. [https://web.njit.edu/~jiang/math614/hestenes-stiefel.pdf](https://web.njit.edu/~jiang/math614/hestenes-stiefel.pdf).

