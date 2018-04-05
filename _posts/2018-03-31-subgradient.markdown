---
layout: post
comments: true
title:  "Subgradient Calculus"
excerpt: "Convex Optimization Part II"
date:   2018-03-31 11:00:00
mathjax: true
---

<!-- 
<svg width="800" height="200">
	<rect width="800" height="200" style="fill:rgb(98,51,20)" />
	<rect width="20" height="50" x="20" y="100" style="fill:rgb(189,106,53)" />
	<rect width="20" height="50" x="760" y="30" style="fill:rgb(77,175,75)" />
	<rect width="10" height="10" x="400" y="60" style="fill:rgb(225,229,224)" />
</svg>
 -->

Now that we've covered topics from convex optimization in a very shallow manner, it's time to go deeper.

## Subgradient methods

### Convexity Review

To understand subgradients, first we need to review convexity. The Taylor series of a real valued function that is infinitely differentiable at a real number \\(x\\) is given by 

$$
f(y) \approx \sum\limits_{k=0}^{\infty} \frac{f^{(k)}(x)}{k!}(y-x)^k
$$

Expanding terms, the series resembles

$$
f(y) \approx  f(x) + \frac{f^{\prime}(x)}{1!}(y-x) + \frac{f^{\prime\prime}(x)}{2!}(y-x)^2 + \frac{f^{\prime\prime\prime}(x)}{3!}(y-x)^3 + \cdots.
$$

If a function is convex, its first order Taylor expansion (a tangent line) will always be a global underestimator:

$$
f(y) \geq f(x) + \frac{f^{\prime}(x)}{1!}(y-x)
$$

If \\(y=x + \delta\\), then:

$$
\begin{aligned}
f(x+ \delta) \geq f(x) + \nabla f(x)^T (x+\delta-x) \\
f(x+ \delta) \geq f(x) + \nabla f(x)^T \delta
\end{aligned}
$$

<div align="center">
<img  src="/assets/tangent_underestimates_quadratic.jpg" width="25%" />
</div>

From the picture, it's obvious that an affine function is always an underestimator for this quadratic (thus convex) function. But there is also intuition behind why this is true. A univariate function is convex if its derivative is increasing (thus second derivative is positive). Since slope is a measure of function increase over some interval \\(\delta\\), where usually \\(\delta=1\\), if we multiply slope by \\(\delta = y-x = \\) "run", then we will be left with only the change in function value over the interval (\\(slope=\frac{rise}{run}\\) and \\(\frac{rise}{run}\cdot(run)=rise\\) ).

If \\(y>x\\), our \\(\delta\\) is positive, and since the derivative is increasing (which an affine function cannot account for), we always underestimate.

If \\(x>y\\), our \\(\delta\\) is negative, and since the ...

## Subgradients

Consider a function \\(f\\) with a kink inside of it, rendering the function non-differentiable at a point \\(x_2\\).

<div align="center">
<img  src="/assets/subgradients.png" width="50%" />
</div>

The derivative jumps up by some amount (over some interval) at this kink. Any slope within that interval is a valid subgradient. A subgradient is also a supporting hyperplane to the epigraph of the function. The set of subgradients of \\(f\\) at \\(x\\) is called the subdifferential \\(\partial f(x)\\). To determine if a function issubdifferentiable at \\(x_0\\), we must ask, "is there a global affine lower bound on this function that is tight at this point?".
- If a function is convex, it has at least one point in the relative interior of the domain. 
- If a function is differentiable at \\(x\\), then \\(\partial f(x) = \{\nabla f(x)\} \\) (a singleton set).

For example, for absolute value, we could say the interval \\( [\nabla f_{\text{left}}(x_0), \nabla f_{\text{right}}(x_0)] \\) is the range of possible slopes (subgradients):

<div align="center">
<img  src="/assets/subdifferential.png" width="50%" />
</div>




A subdifferential is a point-to-set mapping.



### Subgradient methods 

From [1]: Given a convex function \\(f:\mathbb{R}^n \rightarrow \mathbb{R}\\), not necessarily differentiable. Subgradient method is just like gradient descent, but replacing gradients with subgradients. I.e., initialize \\(x^{(0)}\\), then repeat

$$
\begin{array}{ll}
x^{(k)} = x^{(k−1)} − t_k \cdot g^{(k−1)}, & k = 1, 2, 3, \cdots
\end{array}
$$

where \\(g^{(k−1)}\\) is **any** subgradient of \\(f\\) at \\(x^{(k−1)}\\). We keep track of best iterate \\(x_{best}^k\\)
 among \\(x^{(1)}, \cdots , x^{(k)}\\):

$$
f(x_{best}^{(k)}) = \underset{i=1,\cdots ,k}{\mbox{ min  }}  f(x^{(i)})
$$

To update each \\(x^{(i)}\\), there are basically two ways to select the step size:
- Fixed step size: \\(t_k = t\\) for all \\(k = 1, 2, 3, \cdots \\)
- Diminishing step size: choose \\(t_k\\) to satisfy

$$
\begin{array}{ll}
\sum\limits_{k=1}^{\infty} t_k^2 < \infty , & \sum\limits_{k=1}^{\infty} t_k < \infty 
\end{array}
$$

### Subgradient methods for constrained problems

### Primal-dual subgradient methods

### Stochastic subgradient method 

### Mirror descent and variable metric methods

## Localization methods

### Localization and cutting-plane methods

### Analytic center cutting-plane method

### Ellipsoid method

## Decomposition and distributed optimization

### Primal and dual decomposition

### Decomposition applications

### Distributed optimization via circuits

## Proximal and operator splitting methods

### Proximal algorithms

### Monotone operators

### Monotone operator splitting methods

### Alternating direction method of multipliers (ADMM)

## Conjugate gradients

### Conjugate-gradient method

### Truncated Newton methods

## Nonconvex problems

### \\(l_1\\) methods for convex-cardinality problems 

### \\(l_1\\) methods for convex-cardinality problems, part II 

### Sequential convex programming

## Branch-and-bound methods 

## References

[1] Gordon, Geoff. CMU 10-725 Optimization Fall 2012 Lecture Slides, [Lecture 6](https://www.cs.cmu.edu/~ggordon/10725-F12/scribes/10725_Lecture6.pdf). 



