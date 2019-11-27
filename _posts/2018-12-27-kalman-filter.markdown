---
layout: post
title:  "The Kalman Filter"
permalink: /kalman-filter/
excerpt: "Multivariate Gaussians, ..."
mathjax: true
date:   2018-12-27 11:00:00
mathjax: true

---
Table of Contents:
- [What is the Kalman Filter?](#kf-overview)
- [Tracking 3D Bounding Boxes with Constant Velocity](#3d-bbox-constant-velocity)
- [Practical Example: 3D Tracking with ICP](#3dtracking-icp)

<a name='kf-overview'></a>
## What is the Kalman Filter?

The Kalman Filter is nothing more than the Bayes' Filter for Gaussian distributed random variables. The Bayes' Filter is described in a [previous post](/bayes-filter/).

It is a very surprising result that we can write out the integrals analytically for the Bayes' Filter when working with a special family of distributions: Gaussian distributed random variables (r.v.'s).  As we recall from the Bayes' Filter, we have two key recurrences for prediction and updates, respectively:

$$
p(x_{t} \mid y_{1:t-1}) = \int_{x_t-1} p(x_{t} \mid x_{t-1}) p(x_{t-1} \mid y_{1:t-1})dx_{t-1}
$$

$$
p(x_{t} \mid y_{1:t}) = \frac{p(y_{t} \mid x_{t})p(x_{t} \mid y_{1:t-1})}{\int\limits_{x_{t}} p(y_{t} \mid x_{t}) p(x_{t} \mid y_{1:t-1}) dx_{t}}
$$

Note that to compute these two recurrences, there are only three quantities that we'll need to be able to evaluate:
- (1) $$ \int_{x_{t-1}} p(x_t \mid x_{t-1}) p(x_{t-1} \mid y_{1:t-1}) dx_{t-1} $$
- (2) $$ p(x_t \mid x_{t-1})  = f(x_t, x_{t-1} ) $$
- (3) $$ p(y_t \mid x_t) = g(y_t, x_t) $$

Expressions (2) and (3) must be finitely parameterizable. We call (2) the *transition distribution* and we call (3) the *measurement likelihood*.

It will take a fair amount of work to derive the analytical formulas of the Bayes' Filter for Gaussian r.v.'s (the Kalman Filter).  Gaussian distributions are the *only* useful family of distributions for which we can have a closed-form equation for the recursive filter. We'll first review properties of multivariate Gaussians, then the Woodbury matrix inversion lemmas, intuition behind covariance matrices, and then derive the Kalman Filter.





## Woodbury Matrix Inversion Lemmas

## Kalman Filter Derivation





## Jointly Guassian Random Vectors


Suppose $$S = (X,Y)$$ (broken into two pieces)

$$
S = (X,Y) \sim \mathcal{N}(\mu, \Sigma)
$$
where 

$$
\begin{array}{ll}
\mu = \begin{bmatrix} \mu_x \\ \mu_y \end{bmatrix}, & \Sigma = \begin{bmatrix} 
\Sigma_x & \Sigma_{xy} \\ \Sigma_{yx} & \Sigma_y \end{bmatrix}
\end{array}
$$

Ultimately, we would like to compute $$p(x \mid y)$$, the Bayesian estimate of $$X \mid Y$$

\begin{equation}
\begin{aligned}
\mu_x = E[X] \\
\mu_y = E[Y] \\
\Sigma_X = Cov(X) \\
\Sigma_Y = Cov(Y) \\
\Sigma_{XY} = Cov(X,Y) \\
\Sigma_{YX} = Cov(Y,X) = \Sigma_{XY}^T
\end{aligned}
\end{equation}

Claim: The marginals are Gaussian:
\begin{equation}
X \sim p(x) = \int_y p(x,y) dx = \mathcal{N}(\mu_x, \Sigma_x)
\end{equation}
\begin{equation}
Y \sim p(y) = \int_x p(x,y)dx = \mathcal{N}(\mu_y, \Sigma_y)
\end{equation}

## Conditional Gaussian Multivariate Variables

The Conditionals are also Gaussian [see here](https://stats.stackexchange.com/questions/30588/deriving-the-conditional-distributions-of-a-multivariate-normal-distribution):

We will recognize the Schur complement [Boyd, A.5.5](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)


\begin{equation}
X \mid Y \sim p(x \mid y) = \mathcal{N}(\mu_{x\mid y}, \Sigma_{x \mid y})
\end{equation}
We use 
\begin{equation}
\mu_{X \mid Y} = \mu_x + \Sigma_{XY} \Sigma_Y^{-1}(y - \mu_y)
\end{equation}

\begin{equation}
\Sigma_{X \mid Y} = \Sigma_x - \Sigma_{XY} \Sigma_Y^{-1} \Sigma_{YX}
\end{equation}
Same for 
\begin{equation}
Y  \mid X \sim p(y \mid x) = \mathcal{N}(\mu_{y \mid x}, \Sigma_{y \mid x})
\end{equation}
\item We recall that
\begin{equation}
p(x,y) = p(x \mid y) p(y) = p(y \mid x)p(x)
\end{equation}
\item which immediately gives us Bayes Rule
\begin{equation}
p(x \mid y) =  \frac{p(y \mid x)p(x)}{p(y)}
\end{equation}
\item Let's prove that ?????????

$$
\mathcal{N}(\mu, \Sigma) = \mathcal{N}\Bigg(\begin{bmatrix} \mu_x \\ \mu_y \end{bmatrix}, \begin{bmatrix} \Sigma_x & \Sigma_{xy} \\ \Sigma_{yx} & \Sigma_y \end{bmatrix} \Bigg) = \mathcal{N}(\mu_{x\mid y}, \Sigma_{x \mid y}) \mathcal{N}(\mu_{y}, \Sigma_{ y})
$$
???????????????


## A Proof by Demonstration

Working Backwards:

$$
p(x \mid y) p(y) = \mathcal{N}(\mu_{x\mid y}, \Sigma_{x \mid y}) \mathcal{N}(\mu_{y}, \Sigma_{ y})
$$

$$
\mathcal{N}(\mu_x, \Sigma_x) = 
\eta_{x \mid y }\mbox{exp}
\Bigg\{ -\frac{1}{2} (x - \mu_{x \mid y})^T \Sigma_{x \mid y}^{-1} (x - \mu_{x \mid y}) \Bigg\} \cdot \eta_{ y }\mbox{exp}
\Bigg\{ -\frac{1}{2} (y - \mu_y)^T \Sigma_y^{-1} (y - \mu_y) \Bigg\}
$$

Combine exponentials

$$
\mathcal{N}(\mu_x, \Sigma_x) =
\eta_{x \mid y } \eta_{ y } \mbox{ exp}
\Bigg\{ -\frac{1}{2} \begin{bmatrix} (x - \mu_{x \mid y}) \\(y - \mu_y)  \end{bmatrix}^T \begin{bmatrix} \Sigma_{x \mid y}^{-1} & 0 \\ 0 &  \Sigma_y^{-1}  \end{bmatrix} \begin{bmatrix} (x - \mu_{x \mid y}) \\ (y - \mu_y)  \end{bmatrix} \Bigg\} 
$$



We want to show that the exponent becomes:

$$
\begin{bmatrix}
(x - \mu_{x \mid y}) \Sigma_{x \mid y}^{-1} \\
(y - \mu_y)  \Sigma_y^{-1} 
\end{bmatrix}\begin{bmatrix} (x - \mu_{x \mid y}) \\ (y - \mu_y)  \end{bmatrix} 
$$

We have 3 matrices multiplied together

$$
\begin{bmatrix}   \\ (y - \mu_y)  \end{bmatrix}     \begin{bmatrix} & 0 \\ 0 & \Sigma_y^{-1} \end{bmatrix}     \begin{bmatrix}  x - \mu_x - \Sigma_{XY} \Sigma_Y^{-1} (y-\mu_y) \\ (y - \mu_y) \end{bmatrix} 
$$

Notice that

$$
\begin{bmatrix} x - \mu_x - \Sigma_{XY} \Sigma_Y^{-1} (y-\mu_y) \\ (y - \mu_y) \end{bmatrix} = \begin{bmatrix} I & -\Sigma_{XY} \Sigma_Y^{-1} \\ 0 & I \end{bmatrix} \begin{bmatrix} (x- \mu_x) \\ (y - \mu_y) \end{bmatrix}
$$

Now we go from

$$
 = \begin{bmatrix} (x-\mu_x) \\ (y-\mu_y) \end{bmatrix}^T \begin{bmatrix} I & 0 \\  -\Sigma_Y^{-1}\Sigma_{YX} & I \end{bmatrix} \begin{bmatrix} \Bigg( \Sigma_{x} - \Sigma_{xy} \Sigma_{y}^{-1}  \Sigma_{yx}\Bigg)^{-1}  & 0 \\ 0 & \Sigma_{y}^{-1} \end{bmatrix} \begin{bmatrix} I & -\Sigma_{XY} \Sigma_Y^{-1} \\ 0 & I \end{bmatrix} \begin{bmatrix} (x-\mu_x) \\ (y-\mu_y) \end{bmatrix}
$$

to

Since $$LD^{-1}U = \Sigma^{-1}$$ \\

$$D^{-1} = L^{-1}\Sigma^{-1} U^{-1}$$ \\

Triangular matrix: eigenvalues on diagonal. Here eigenvalues are all along diagonal, and are all 1, so this must be invertible.

$$D = U \Sigma L$$ \\

$$\iff p(x \mid y) p(y) = p(x,y)$$


$$
 = \begin{bmatrix} (x-\mu_x) \\ (y-\mu_y) \end{bmatrix}^T  \begin{bmatrix} \Sigma_{x} & \Sigma_{xy} \\ \Sigma_{yx} & \Sigma_{y} \end{bmatrix}^{-1} \begin{bmatrix} (x-\mu_x) \\ (y-\mu_y) \end{bmatrix} =   \begin{bmatrix} (x-\mu_x) \\ (y-\mu_y) \end{bmatrix}^T  \Sigma^{-1} \begin{bmatrix} (x-\mu_x) \\ (y-\mu_y) \end{bmatrix}
$$

$$
\Sigma L = 
\begin{bmatrix} \Sigma_{x} & \Sigma_{xy} \\ \Sigma_{yx} & \Sigma_{y} \end{bmatrix} \begin{bmatrix} I & 0 \\  -\Sigma_Y^{-1}\Sigma_{YX} & I \end{bmatrix}
$$



$$
U \Sigma L = \begin{bmatrix} I & -\Sigma_{xy} \Sigma_{y}^{-1} \\ 0 & I \end{bmatrix} \begin{bmatrix} \Sigma_{x \mid y} & \Sigma_{xy} \\ 0 & \Sigma_y \end{bmatrix}
$$

$$
\begin{bmatrix}  \end{bmatrix} \begin{bmatrix} \end{bmatrix} = \begin{bmatrix} \Sigma_{x \mid y} & 0 \\ 0 & \Sigma_{y} \end{bmatrix} = D
$$



## Gaussian Estimation (Bayes' Rule for Gaussian R.V.s) and Recursive Gaussian Estimation (Bayesian Estimation for Gaussian R.V.s)


For the constants
\begin{equation}
\eta_{x \mid y} \eta_y  = \eta_{xy}
\end{equation}

$$
\begin{array}{lll}
\eta_y = \frac{1}{\sqrt{(2\pi)^m |\Sigma_y|}}, & \eta_{x \mid y} = \frac{1}{\sqrt{(2\pi)^n |\Sigma_{x \mid y}|}}, & \eta_{xy} = \frac{1}{\sqrt{(2\pi)^{n+m} |\Sigma|}},
\end{array}
$$

where $$Y \in \mathbb{R}^m, X \in \mathbb{R}^n, (X,Y) \in \mathbb{R}^{n+m}$$

Determinant Identities:
\begin{equation}
|A B| = |A| |B|
\end{equation}

$$
|\begin{bmatrix} A & 0 \\ 0 & B \end{bmatrix}| = |A| |B|
$$


$$
| \begin{bmatrix} I & 0 \\ A & I \end{bmatrix} | = \begin{bmatrix} I & A \\ 0 & I \end{bmatrix} | = 1
$$

We hope to find that

\begin{equation}
\eta_{x \mid y}\eta_y = \frac{1}{(2\pi)^{m/2}(2\pi)^{n/2} |\Sigma_y|^{1/2} |\Sigma_{x \mid y}|^{1/2} } =  \frac{1}{(2\pi)^{ \frac{m+n}{2} }|\Sigma|^{1/2} }
\end{equation}

Now we use the fact that
\begin{equation}
D = U \Sigma L 
\end{equation}
and we define
\begin{equation}
D = \begin{bmatrix} \Sigma_{x \mid y} & 0 \\ 0 & \Sigma_y  \end{bmatrix}
\end{equation}

By independence,
\begin{equation}
\begin{aligned}
p(y) = \int_x p(x,y) dx \\
\int_x \tilde{p}(x \mid y) \tilde{p}(y) dx \\
\int_x \tilde{p}(x \mid y) dx \tilde{p}(y) \\
= \tilde{p}(y)
\end{aligned}
\end{equation}


We recall that a conditional multivariate Gaussian distribution $$X \mid Y$$ has a pdf parameterized as follows:

\begin{equation}
p(x \mid y) = \mathcal{N}(\mu_{x\mid y}, \Sigma_{x \mid y}) = \mathcal{N} \Bigg(\mu_x + \Sigma_{xy}\Sigma_{yy}^{-1}(y-\mu_y),  \Sigma_{xx} - \Sigma_{xy} \Sigma_{yy}^{-1}\Sigma_{yx} \Bigg)
\end{equation}
$\mu_x, \Sigma_x$ are the priors.

if $$\Sigma_y$$ is small, then inverse is huge, reduce uncertainty a lot...
\item Measurement reduces uncertainty in $$X$$ proportionally to $$\Sigma_y^{-1}$$

Never do worse than your prior -- even crappy measurement is good. Shrink error ellipsoid as you get a measurement. We trust it by amount weighted by inverse of our confidence in the measurement. Mostly keep prior if high amount of uncertainty (so add very little, because $$\Sigma^{-1}$$ is small). Not realistic to know the joint distribution:

$$
\begin{array}{ll}
\Sigma  = \begin{bmatrix} \Sigma_x & \Sigma_{xy} \\ \Sigma_{yx} & \Sigma_y  \end{bmatrix}, \mu = \begin{bmatrix} \mu_x \\ \mu_y \end{bmatrix}
\end{array}
$$

Bayes Rule says:

\begin{equation}
p(x \mid y) = \frac{p(y \mid x) p(x)}{ \int_x p(y \mid x)p(x) dx}
\end{equation}
We do know
\begin{equation}
\begin{array}{ll}
\Sigma_{y \mid x}, & \mu_{y \mid x}
\end{array}
\end{equation}

where
\begin{equation}
p(y \mid x) = \mathcal{N}( \mu_{y \mid x}, \Sigma_{y \mid x})
\end{equation}
\item Definition:
\begin{equation}
(X,Y) \sim \mathcal{N}(\mu, \Sigma)
\end{equation}

Theorem: $$(X,Y)$$ are jointly Gaussian iff $$\exists C,M$$ and r.v. $$V,N$$ such that
\begin{equation}
\begin{array}{ll}
Y = CX + V, & X = MY + N
\end{array}
\end{equation}
where $$ C \in \mathbb{R}^{m \times n}$$ and $$M \in \mathbb{R}^{n \times m}$$
\begin{equation}
\begin{aligned}
V \sim \mathcal{N}(\mu_v, R) \\
N \sim \mathcal{N}(\mu_N, \Sigma_N) \\
Cov(X,V) = 0 \\
Cov(Y,N) = 0
\end{aligned}
\end{equation}

Gaussian property is deeply linked to linearity!

Projection of vectors onto different directions (Gaussian random variable can be projected)

Project $$Y$$ into the direction of $X$ (this is what $CX$ does)

$$V$$ is the leftover part of $Y$ that is orthogonal to $X$

Orthogonality of Gaussian R.V.s: covariance is zero. No way to use one to predict the other (can derive formally as inner product in vector space)


## Linear Control Systems

\begin{equation}
\begin{aligned}
\dot{x} = Ax + Bu \\
y = Cx + Du
\end{aligned}
\end{equation}

We use the discrete-timve version in this class, not the continuous form

\begin{equation}
\begin{aligned}
x_{t+1} = Ax_t + \_ + w_t \\ 
y_t = Cx_t + \_ + v_t
\end{aligned}
\end{equation}
where $w_t, v_t$ are noise
\item Expectation is a linear operator: $Y=CX + V$




\item mean:
\begin{equation}
\mu_y = E[Y] = \int_y y \mbox{ } p(y) dy
\end{equation}
where $Y \in \mathbb{R}^n, \mu_y \in \mathbb{R}^n$

\item Covariance:
\begin{equation}
\Sigma_y = E[(Y-\mu_y)(Y-\mu_y)^T] = \int_y (y-\mu_y)(y-\mu_y)^T p(y) dy
\end{equation}
\item 

\begin{equation}
\begin{aligned}
\mathbb{E}[Y] = \mathbb{E}[CX+V] \\
 = \int_{x,v}(Cx+ v) p(x,v) dx dv \\
 = \int_{x,v} Cx \mbox{ } p(x,v) dx dv + \int_{x,v}v \mbox{ } p(x,v) dx dv \\
 = C \int_{x,v} x \mbox{ } p(x,v) dx dv + \int_{x,v} v \mbox{ } p(x,v) dx dv \\
= C \mathbb{E}[X] + \mathbb{E}[V]
\end{aligned}
\end{equation}
so we have shown that $\mu_y = C \mu_x + \mu_v$. $\mu_x$ is a known prior. $C$ is known (from the measruement model), and $\mu_v$ is also known (from the measurement model)
\item 

We can do this because 
\begin{equation}
\begin{aligned}
\int_v v \Bigg( \int_x p(x,v) dx \Bigg) dv
\int_v v \mbox{ } p(v) dv \\ 
= \mathbb{E}[V]
\end{aligned}
\end{equation}
\item Take that $Y = CX + V$: 
\begin{equation}
\begin{aligned}
\Sigma_y = Cov(Y) = \mathbb{E}[(Y-\mu_y)(Y-\mu_y)^T] \\
= \mathbb{E}[((CX + V)-\mu_y)((CX + V)-\mu_y)^T] \\
 = \mathbb{E}[((CX + V)- (C \mu_x + \mu_v) )((CX + V)- (C \mu_x + \mu_v) )^T]  \\
 confusing outer product step \\
  = \mathbb{E}[(V - \mu_v )(CX - C \mu_x)^T ] + \mathbb{E}[(V - \mu_v )(V - \mu_v)^T ]  \\
  = \mathbb{E}[ C(X - \mu_x )(X - \mu_x)^T C^T] +  + \mathbb{E}[(V - \mu_v )(X - \mu_x)^T C^T ]  + \mathbb{E}[(V- \mu_v)(V- \mu_y)^T] 
\end{aligned}
\end{equation}
We know $Cov(V) = \Sigma_v = R$, the measurement noise covariance
\item The covaraince of the prior is known
\begin{equation}
\begin{aligned}
Cov(X) = \Sigma_x \\
 = C \mathbb{E}[(X-\mu_x)(X - \mu_x)^T]C^T \\
 +  C \mathbb{E}[(X-\mu_x)(V - \mu_v)^T] \\
 +  \mathbb{E}[(V-\mu_v)(X - \mu_x)^T]C^T + R \\
\end{aligned}
\end{equation}
where $Cov(V,X) = \Sigma_{vx}$
\item So we find
\begin{equation}
\Sigma_y = C \Sigma_x C^T + C \Sigma_{xv} + \Sigma_{vx} C^T + R
\end{equation}
covariances are zero $Cov(X,V) = 0$, so we get

\begin{equation}
\Sigma_y = C \Sigma_x C^T + R
\end{equation}
where $\Sigma_x$ is the prior covaraince, and the other threee terms $C,C^T,R$ are known from the measurement model
\item so we find for

\begin{equation}
\begin{aligned}
 \Sigma_{xy} = \mathbb{E}[(X-\mu_x)(Y - \mu_y)^T] \\
 = \mathbb{E}[(X-\mu_x)(CX+V - C\mu_x -\mu_v) ] \\
 =   \mathbb{E}[(X-\mu_x)(X - \mu_x)^T]C^T + \mathbb{E}[(X-\mu_x)(V-\mu_v)] \\
 \Sigma_{xy} = \Sigma_x C^T + \Sigma_{xv} 
\end{aligned}
\end{equation}

for $$(X,Y)$$ jointly Gaussian $$\Sigma{xv} = 0$$, we have
\begin{equation}
\Sigma_{xy} = \Sigma_x C^T
\end{equation}

## Bayes Rule for Jointly Gaussian Random Vectors

\begin{equation}
\mu_{x \mid y} = \mu_x + \Sigma_X C^T \Bigg( C \Sigma_x C^T + R)^{-1} (y - [C \mu_x + \mu_v] \Bigg)
\end{equation}
and 
\begin{equation}
\Sigma_{X \mid Y}  = \Sigma_X - \Sigma_X C^T \Bigg( C \Sigma_x C^T + R\Bigg)^{-1} C \Sigma_x
\end{equation}

\item find out $C$ by shaking around sensor (accelerometer). This is a model of our sensor. We could get it in the lab. We could get the measurement likelihood model from the following equation:
\begin{equation}
Y = CX + V
\end{equation}
with $V \sim \mathcal{N}(\mu_v, R)$ and $Cov(X,V) = 0$ 
\item This is equivalent to $p(y \mid x)$, the measurement likelihood
\item $Y=CX+V$ is guaranteed because these two variables are JOINTLY gaussian. Can see this as a projection
\item We know prior: $(\mu_x, \Sigma_x)$
\item From the measurement equation, we know $C, \mu_v, R$

## Kalman Filters

What is the connection between $$Y=CX+V$$ and $$p(y \mid x)$$?

$$p(y \mid x) \sim \mathcal{N}(\mu_{y \mid x}, \Sigma_{y \mid x})$$

$$
\begin{aligned}
\mu_{y \mid x } = \mathbb{E}[Y \mid X=x] \\= \mathbb{E}[CX + V \mid X=x] \\= \mathbb{E}[CX \mid X=x] + \mathbb{E}[V \mid X=x]  \\=  \mathbb{E}[CX \mid X=x] + \mathbb{E}[V ]  \\
\mu_{y \mid x } = Cx + \mu_v
\end{aligned}
$$

because uncorrelated means independent for Gaussians

$$
\begin{aligned}
\Sigma_{Y \mid X} = \mathbb{E}\Bigg[ (Y-\mu_{y \mid x}) (Y-\mu_{y \mid x})^T \mid X=x \Bigg] \\
= \mathbb{E}\Bigg[ (CX+V-\mu_{y \mid x}) (CX+V-\mu_{y \mid x})^T  \mid X=x \Bigg] \\
= \mathbb{E}\Bigg[ (CX+V-Cx - \mu_v) (CX+V-CX - \mu_v )^T  \mid X=x \Bigg] \\
= R
\end{aligned}
$$
So measurement equation encodes the measurement likelihood

When given $$x$$, so lock down in $$p(y \mid x)$$, all randomness comes from noise $$V$$

Since $$(X,Y) \sim \mathcal{N}(\mu, \Sigma)$$, then $$\exists M \in \mathbb{R}^{n \times n}, N \sim \mathcal{N}(\mu_N, \Sigma_N)$$ s.t.
\begin{equation}
\begin{array}{ll}
X = MY + N, & Cov(Y,N) = 0
\end{array}
\end{equation}

We don't know these things though! But the equation is still valid. Linear in $$Y$$, with the following $$M$$

Why not compute
\begin{equation}
p(x \mid y) = \mathcal{N}\Big(  \mu_{x \mid y}, \Sigma_{x \mid y}\Big)
\end{equation}
where $M = \Sigma_x C^T (C \Sigma_x C^T + R)^{-1}$

where $\mu+N$ is
\begin{equation}
\mu_N  = \mu_x - (\Sigma_x \cdots) (C \mu_x - \mu_v)
\end{equation}
and
\begin{equation}
\Sigma_{x \mid y} = \Sigma_N
\end{equation}
and $\mu_{x \mid y} = My + \mu_N$


## Recursive Bayesian Estimation from Jointly Gaussian R.V.s with conditionally independent measurements, Treat Output as Input to next time step

Naive Bayes model of measurements

\begin{equation}
\mu_{x \mid y_{1:t} } = \mu_{x \mid y_{1:t-1} } + \Sigma_{x  \mid y_{1:t-1}} C^T (C \Sigma_{X \mid Y_{1:t-1}} C^T + R)^{-1} \Bigg( y_t - ( C \mu_{x \mid y_{1:t-1}} + \mu_v) \Bigg)
\end{equation}
\item 
\begin{equation}
\Sigma_{X \mid Y_{1:t} }  = \Sigma_{X \mid Y_{1:t-1} } - \Sigma_{X \mid Y_{1:t-1} } C^T \Bigg( C \Sigma_{X \mid Y_{1:t-1} } C^T + R\Bigg)^{-1} C \Sigma_{X \mid Y_{1:t-1} }
\end{equation}
\item Fuse uncertainty in the correct way
\item Covariance should always decrease (but it is a matrix)
\item For a matrix to decrease, we mean that the differencew between the matrix is definite (negative or positive definite)
\item To decrease -- expect difference to be positive semidefinite
\begin{equation}
(\Sigma_{x \mid Y_{1:t-1}} - \Sigma_{x \mid y_{1:t}}) \geq 0
\end{equation}

Measurements have to be conditionally independent. Suppose each measurement has a corresponding equation:
\begin{equation}
Y_t = C_t X + V_t
\end{equation}

We can stack all of the measurements into one block, as follows
\begin{equation}
\begin{bmatrix}
Y_1 \\ \vdots \\ Y_t
\end{bmatrix} = \begin{bmatrix} C_1 \\ \vdots \\ C_t  \end{bmatrix} X + \begin{bmatrix}  V_1 \\ \vdots \\ V_t \end{bmatrix}
\end{equation}

Being able to break up the measurements is special

Consider two measurments at $$Y_t, Y_{\tau}$$. We wish to compute
\begin{equation}
Cov(Y_t, Y_{\tau} \mid X)
\end{equation}
\item We already know that
\begin{equation}
\begin{aligned}
\mu_{Y_t \mid x} = C_t X + \mu_{v_t} \\
\mu_{Y_{\tau} \mid x} = C_{\tau} X + \mu_{v_{\tau}} \\
\end{aligned}
\end{equation}
Conditional covariance

\begin{equation}
Cov(Y_t, Y_{\tau} \mid X) = \mathbb{E}\Bigg[ (Y_t - \mu_{Y_t \mid x})(Y_{\tau} - \mu_{Y_{\tau} \mid x})^T \mid X \Bigg]
\end{equation}

\begin{equation}
Cov(Y_t, Y_{\tau} \mid X) = \mathbb{E}\Bigg[ (Y_t - C_t X - \mu_{v_t})(Y_{\tau} -  C_{\tau} X - \mu_{v_{\tau}})^T \mid X \Bigg]
\end{equation}

With the definition of $Y_t, Y_{\tau}$:
\begin{equation}
Cov(Y_t, Y_{\tau} \mid X) = \mathbb{E}\Bigg[ (C_t X + V_t - C_t X - \mu_{v_t})( C_{\tau} X + V_{\tau} -  C_{\tau} X - \mu_{v_{\tau}})^T \mid X \Bigg]
\end{equation}
Canceling terms
\begin{equation}
Cov(Y_t, Y_{\tau} \mid X) = \mathbb{E}\Bigg[ ( V_t - \mu_{v_t})( V_{\tau} -   \mu_{v_{\tau}})^T \mid X \Bigg]
\end{equation}
And conditionally independent of $X$ because WHY???
\begin{equation}
Cov(Y_t, Y_{\tau} \mid X) = \mathbb{E}\Bigg[ ( V_t - \mu_{v_t})( V_{\tau} -   \mu_{v_{\tau}})^T  \Bigg]
\end{equation}

Uncorrelated measurement noise! 
\item White noise: sequence of measurements $(v_1, v_2, \dots, v_t)$ s.t. $Cov(v_t, v_{\tau} = 0, \forall t \neq \tau$
\end{itemize}

## Gaussian White Noise Process


\begin{equation}
\begin{aligned}
V_t \sim GWP(\mu_{v_t}, R_t) \\
V_t \sim \mathcal{N}( \mu_{v_t}, R_t) \\
Cov(V_t, V_{\tau}) = 0
\end{aligned}
\end{equation}

Flipping around -- white noise process, knowing all of the previous noise, won't help you predict anything in the future

Otherwise, get Delta Dirac Function, with R at $$t-\tau=0$$. We hope coloring is very small, compared to the other dynamics in your process (coloring looks like a Laplacian distribution). We hope to put as much of the sensor into the state as possible
\item White noise: sucked out everything from the sensor measurement that relates to the thing that you are measuring
\item White noise in measurment is essential to enforcing the conditional independence assumption!!! To use a linear estimator with a constant state...????
\end{itemize}
\subsection{Kalman Filter}
\begin{itemize}
\item Bayeisan filter from Linear-Gaussian system, with Markov prooperty and conditionally independent measurements
\item Linear-Gausssian state-space system
\item By the dynamics equation,
\begin{equation}
X_{t+1} = A_t x_t + B u_t + W_t
\end{equation}
By the measurement equation,
\begin{equation}
Y_t = C_t X_t + D_t u_t + V_t
\end{equation}
\item $V_t \sim GWP(0, R_t)$
\item Where $D_tu_t = \mu_v$, the mean of our \textbf{sensor noise}
\item We also need our \textbf{process noise} to be white:
\begin{equation}
W_t \sim GWP(0, Q_t)
\end{equation}
\item And uncorrelated with the initial state:
\begin{equation}
\begin{aligned}
Cov(X_0, V_t) = 0, \forall t\\
Cov(x_0, W_t) = 0, \forall t\\
Cov(w_t,v_{\tau}) = 0, \forall \tau,t
\end{aligned}
\end{equation}
I.C.
\begin{equation}
X_0 \sim \mathcal{N}(\mu_0,\Sigma_0)
\end{equation}
Must be HMM undirected model, which gives us two kinds of conditional independence: \\
(1) Markov -  $p(x_{t+1} \mid x_t, x_{0:t-1},y_{1:t}) = p(x_{t+1} \mid x_t)$ \\
(2) Cond. Independence of Measurment -- 
\begin{equation}
p(y_t \mid x_{0:t},y_{1:t-1}) = p(y_t \mid x_t)
\end{equation}

\end{itemize}
\subsection{Kalman Filter Equations}
\begin{itemize}
\item Notation: 

\begin{equation}
\begin{aligned}
\mu_{t \mid t} = \mathbb{E}[X_t \mid Y_{1:t}] \\
\Sigma_{t \mid t} = Cov(X_t \mid Y_{1:t}) \\
\mu_{t \mid t-1} = \mathbb{E}[X_t \mid Y_{1:t-1}] \\
\Sigma_{t \mid t-1} = Cov(X_t \mid Y_{1:t-1})
\end{aligned}
\end{equation}

Predict Step:

\begin{equation}
\begin{aligned}
\mu_{t \mid t-1} = A_{t-1} \mu_{t-1 \mid t-1} + B_{t-1} u_{t-1} \\
\Sigma_{t \mid t-1} = A_{t-1} \Sigma_{t-1 \mid t-1} A_{t-1}^T + Q_{t-1}
\end{aligned}
\end{equation}

Update Step:

$$
\begin{aligned}
\mu_{t \mid t} &= \mu_{t \mid t-1} + \Sigma_{t \mid t-1} C_t^T (C_t \Sigma_{t \mid t-1} C_t^T + R_t)^{-1} \Bigg( y_t - (C_t \mu_{t \mid t-1} + D_t u_{t}) \Bigg) \\
\Sigma_{t \mid t} &= \Sigma_{t \mid t-1}  - \Sigma_{t \mid t-1} C_t^T\Bigg(C_t \Sigma_{t \mid t-1} C_t^T + R_t \Bigg)^{-1} C_t \Sigma_{t \mid t-1}
\end{aligned}
$$

## Kalman Filter Continued


Find a model for $$X_{t+1} = f(X_{0:t}, Y_{1:t})$$ and find requirements on $$f(\cdot)$$ s.t. Markov
\begin{equation}
(X_{0:t+1},Y_{1:t}) \sim \mathcal{N}(\mu, \Sigma)
\end{equation}
where
\begin{equation}
(X,Y) \sim \mathcal{N} \
\end{equation}
where 

\begin{equation}
\begin{array}{ll}
Y = CX + V, & Cov(X,V) = 0
\end{array}
\end{equation}

We write equations to slice a vector into pieces

\begin{equation}
\begin{aligned}
X_{t+1} = M(X_{0:t},Y_{1:t}) + W_t \\
X_{t+1} = A_tX_t + \sum\limits_{\tau=0}^{t-1}\Phi_{\tau}X_{\tau} + \sum\limits_{\tau=0}^t \theta_{\tau} Y_{\tau}  + W_t
\end{aligned}
\end{equation}

where we 

\begin{equation}
\begin{aligned}
Cov(X_{\tau}, W_t) = 0, \forall \tau \leq t \\
Cov(Y_{\tau}, W_t) = 0, \forall \tau \leq t \\
\end{aligned}
\end{equation}

By Markov Property, the rest of the matrices must be zero!
\begin{equation}
\mathbb{E}\Bigg[ X_{t+1} \mid X_t, X_{0:t-1}, Y_{1:t}\Bigg] = \mathbb{E}[X_{t+1} \mid X_t]
\end{equation}

So all of the terms in the sum that involve $$\Phi_{\tau}=0 \forall \tau \leq t-1$$ and also $$\Theta_{\tau} = 0, \forall \tau \leq t$$
must equal zero

\begin{equation}
\begin{aligned}
X_{t+1} = A_tX_t + W_t \\
Cov(X_{\tau},W_t) = 0, \forall \tau \leq t, \\
Cov( Y_{\tau}, W_t) = 0, \forall \tau \leq t
\end{aligned}
\end{equation}

Now we can nest things, apply the recursion once backwards
\begin{equation}
\begin{aligned}
X_{t+1} = A_tX_t + W_t \\
X_{t+1} = A_t (A_{t-1} X_{t-1} + W_{t-1}) + W_t\\
X_{t+1} = A_t (A_{t-1} (A_{t-2} X_t W_t) + W_t) + W_t\\
X_{t+1} =\Big( A_t A_{t-1} A_{t-2}\Big) X_{t-2} + ( A_t (A_{t-1} \Big(A_{t-2})W_{t-2}\Bigg) + W_t\Big) + A_tW_{t-1}) + W_t\\
X_{t+1} = \Bigg( \prod\limits_{\tau=0}^t A_{\tau}  \Bigg)X_0 + \sum\limits_{\tau=1}^t  (\prod\limits_{i=\tau}^t A_i) W_{\tau-1} + W_t \\
X_{t+1} = X_{t+1}(X_0, W_{0:t})
\end{aligned}
\end{equation}
deterministically
\begin{equation}
X_{\tau}(X_0, W_{0:\tau-1})
\end{equation}
As long as noise is independent that everything that $X_{\tau}$ depends on, then it will be independent of $X_{\tau}$.

We call Gaussian white noise process.

Projection onto another:
\begin{equation}
\begin{aligned}
Y_t = C_tX_t + \sum\limits_{\tau=0}^{t-1}\Psi_{\tau}X_{\tau} + \sum\limits_{\tau=0}^t \Xi_{\tau} Y_{\tau}  + V_t
\end{aligned}
\end{equation}

We want conditionally independent measurements. So all terms involving $\Psi_{\tau}, \Xi_{\tau}$ are zero.

\begin{equation}
\begin{aligned}
Y_t = C_tX_t + V_t \\
Cov(X_{\tau}, V_t) = 0, \forall \tau \leq t\\
Cov(Y_{\tau}, V_t) = 0, \forall \tau < t
\end{aligned}
\end{equation}
We recall that $X_{\tau}(X_0, W_{0:\tau-1})$.\\
Since $Y_t = C_tX_t + V_t \implies Y_{\tau}(X_0, W_{0:\tau-1}, V_{\tau})$.

\end{itemize}
\subsection{To fulfill the other Covariance=0 reqs}
\begin{itemize}
\item If $Cov(X_{\tau}, V_t) = 0, \forall \tau \leq t$, then $\implies$:\\
\begin{equation}
\begin{aligned}
Cov() =  \\
Cov() = 
\end{aligned}
\end{equation}

So we have white noise. We just derived the dynamics equaiton (without the input $B_tu_t$ term). These where the dirty details of the linear Gaussian model.

\end{itemize}
\subsection{Next Steps}
\begin{itemize}
\item Bayesian Filter, but in Gaussian step. Basic rules to update $\mu, \Sigma$. \\
Predict:
\begin{equation}
p(x_{t+1} | y_{1:t})  \sim \mathcal{N}(\mu_{t+1 \mid t}, \Sigma_{t+1 \mid t})
\end{equation}
Now, the mean:
\begin{equation}
\mu_{t+1 \mid t} = \mathbb{E}[ X_{t+1} \mid Y_{1:t}] = \mathbb{E}[A_tX_t + W_t \mid Y_{1:t}]
\end{equation}
which is 
\begin{equation}
=A_t \mathbb{E}[X_t \mid Y_{1:t}] + \mathbb{E}[W_t \mid Y_{1:t}]
\end{equation}
where the first term is $\mu_{t \mid t}$, and the second term is $\mathbb{E}[W_t]$.

And we know that $Cov(Y_{\tau}, W_t) = 0, \forall \tau \leq t$.

This noise has a mean! 
\begin{equation}
\mathbb{E}[\tilde{w}_t] = \mu_{w_t} = B_t u_t
\end{equation}
We define
\begin{equation}
w_t = \tilde{w_t} - B_tu_t
\end{equation}
where
\begin{equation}
w_t \sim \mathcal{N}(0, Q_t)
\end{equation}
What have we shown, we have shown VERY IMPORTANT
\begin{equation}
\mu_{t+1 \mid t} = A_t \mu_{t \mid t} + B_t u_t
\end{equation}

\end{itemize}
\subsection{Now for the covariance}
\begin{itemize}
\item  Control input does not show up in error covariance

\begin{equation}
\Sigma_{t+1 \mid t} = \mathbb{E}[ (X_{t+1} - \mu_{t+1 \mid t}) (X_{t+1} - \mu_{t+1 \mid t}) \mid Y_{1:t}]
\end{equation}
Pluggin in $X_{t+1} = A_tX_t + B u_t + W_t $ definition, and that $\mu_{t+1 \mid t} = A_t \mu_{t \mid t} + B_tu_t$, we can see
\begin{equation}
\Sigma_{t+1 \mid t} = \mathbb{E}[ ( A_tX_t + B u_t + W_t  - (A_t \mu_{t \mid t} + B_tu_t) ) ( A_tX_t + B u_t + W_t - (A_t \mu_{t \mid t} + B_tu_t) ) \mid Y_{1:t}]
\end{equation}
the $B u_t$'s cancel, so we have\\

TYPE UP THE REST FROM THE PHOTO\\

Leveraging conditional independences, we have shown in the problem set that
\begin{equation}
\mathbb{E}[ (X_t - \mu_{t \mid t}) W_t^T \mid Y_{1:t}] = 0
\end{equation}
We end up with the very important EQUATION
\begin{equation}
\Sigma_{t+1 \mid t} = A_t \Sigma_{t \mid t} A_t^T + Q_t
\end{equation}


## Update Step for KF
We remember

\begin{equation}
p(x_t \mid y_{1:t})   = \frac{ p(y_t \mid x_t) p(x_t \mid y_{1: t-1}) }{\int_{x_t} \cdots dx_t}
\end{equation}

ADD THE NOTES FROM THE PICTURE I TOOK\\

The expected value of $$Y$$, given previous, is

\begin{equation}
\hat{y}_{t \mid t-1} = \mathbb{E}[Y_t \mid Y_{1:t-1}] = \mathbb{E}[C_tX_t + D_t u_t \mid Y_{1:t-1}] = 
\end{equation}
So we subtract this out, $$(y - \hat{y}_{t \mid t-1} )$$.

\item You predicted your mean, so try to correct now! Here is what we thought the measurement should have been! The innovation is the disagreement between the two. This is the innovation process.

The Kalman Gain! is what you add on.


## Kalman and EKF


In the Kalman Filter, need to be given initial $$\mu_{0 \mid 0}, \Sigma_{0 \mid 0}$$, where $$X_0 \sim \mathcal{N}(\mu_{0 \mid 0}, \Sigma_{0 \mid 0}$$


By the dynamics equation,

\begin{equation}
X_{t+1} = A_t x_t + B u_t + W_t
\end{equation}

By the measurement equation,

\begin{equation}
Y_t = C_t X_t + D_t u_t + V_t
\end{equation}

Almost always, $$D_t = 0$$ so that you don't mix up your actuators and your sensors
\item E.g. single integrator particle
\item If you have a filter, you have to stop the filter somewhere, and the computation begins
\item Manually discretize the dynamics, being aware that we ARE GOING AWAY FROM PHYSICS, INTO COMPUTATION
\item dynamics: $$\dot{x} = u$$
\item measurement: $$y = x$$
\item First order Euler discretization. Assume rate of change is constant

\begin{equation}
x_{t+1} = x_t + \delta t u_t
\end{equation}

where $$\delta t = 1$$
Works for relatively well behaved dynamics. We introduce some noise:

\begin{equation}
\begin{aligned}
x_{t+1} = x_t + u_t \\
y_t = x_t
\end{aligned}
\end{equation}

Process noise

\begin{equation}
\begin{aligned}
W_t \sim \mathcal{N}(0,1), process\_ noise \\
V_t \sim \mathcal{N}(0,1), measurement\_ noise
\end{aligned}
\end{equation}

\item I.C. $X_0 \sim \mathcal{N}(0,1)$
\item Linear Gaussian System:
\begin{equation}
\begin{aligned}
X_{t+1} = X_t + u_t + W_t\\
Y_t = X_t + V_t
\end{aligned}
\end{equation}

\item K.F. says
\begin{equation}
\begin{aligned}
\mu_{t+1 \mid t} = \mu_{t \mid t} + u_t \\
\mu_{t\mid t} = \mu_{t \mid t-1} + \Sigma_{t \mid t-1} ( \Sigma_{t \mid t-1})^{-1} (y_t - \mu_{t \mid t-1})
\end{aligned}
\end{equation}
\item $\mu$ will bounce around in a noisy fashion
\item Covariance are completely independent from the specific data acquired by the system. Can solve for it offline. Know how confident you will be, without having taken any data (Not true for EKF or UKF, just for linear system) 
\item Very weird property! Just baked into the linearity of the model
\item Covariance tells you how confident you are
\item If covaraince is huge, crappy estimate
\item Often as interested in covaraince, as you are in the mean

\item Consider example: 

\begin{equation}
\begin{aligned}
A_t = 1, B_t = 1, Q_T = 1 \\
C_t = 1, D_t = 0, R_t = 1
\end{aligned}
\end{equation}
\item Evolution of $\Sigma_{t \mid t}$:
\begin{equation}
\Sigma_{t \mid t} = \Sigma_{t \mid t-1 } -\Sigma_{t \mid t -1} ( \Sigma_{t \mid t-1} + 1)^{-1} \Sigma_{t \mid t-1}
\end{equation}



\item Here the state is a scalar, so $\Sigma$ is a scalar
\begin{equation}
\Sigma_{t \mid t} = \frac{\Sigma_{t \mid t -1} ( \Sigma_{t \mid t -1} + 1) - \Sigma_{t \mid t -1}^2}{\Sigma_{t \mid t -1} + 1}
\end{equation}
\item So the covariance goes down in every single step! $ \Sigma_{t \mid t } \leq \Sigma_{t \mid t -1}$
\item So in our update step, 
\begin{equation}
\Sigma_{t \mid t} = \frac{\Sigma_{t \mid t - 1}}{ \Sigma_{t \mid t-1} + 1}
\end{equation}








## Gaussian White Noise

In practice, noise is usually not Gaussian and is not white (usually colored).

 \item Fact: For a Linear-Gaussian dynamical system (one who's dynamics are expressed with Gaussian White Noise
\begin{equation}
\begin{array}{ll}
x_{t+1} = Ax_t + B u_t + w_t, & w_t \sim \mathcal{N}(0,Q) \\
\end{array}
\end{equation}

\begin{equation}
\begin{array}{ll}
y_t = Cx_t + Du_t + v_t, & v_t \sim \mathcal{N}(O, R)
\end{array}
\end{equation}
where $w_t, v_t$ are zero-mean white noise


## EKF



## UKF


<a name='3d-bbox-constant-velocity'></a>
## Practical Example: Tracking 3D Bounding Boxes with Constant Velocity

Consider the following scenario: we use a deep network to estimate 3d bounding boxes as we drive through a city. At each timestep $$t$$, we receive a set of $$n_t$$ 3d object detections $$D_t = \{D_t^1, D_t^2, \cdots, D_t^{n_t} \}$$. We wish to track objects in 3D through time. 

Weng and Kitani introduce a simple baseline for 3d tracking using Kalman Filters in [1]. They also provide Python code [here](https://github.com/xinshuoweng/AB3DMOT) for a 3D multi-object tracking baseline.

A 3d bounding box detection $$D_t^i$$ is modeled as an 8-tuple $$(x, y, z, l, w, h, \theta, s)$$. These 8 parameters represent the 3D coordinate of the object center $$(x, y, z)$$, the object’s size $$(l, w, h)$$, heading angle $$\theta$$ and its confidence $$s$$. 

The state of a tracked object trajectory is modeled as a 10-dimensional vector $$ \mathbf{x} = (x, y, z, \theta, l, w, h, v_x, v_y, v_z)$$

A constant velocity model is used to propogate tracks from frame $$t$$ to frame $$(t+1)$$:

$$
\begin{aligned}
x_{est} &= x + v_x, \\ 
y_{est} &= y + v_y, \\
 z_{est} &= z + v_z
 \end{aligned}
$$

The state transition matrix $$F \in \mathbb{R}^{n \times n}$$ is thus a $$10 \times 10$$ matrix that we use to accomplish this motion model:

$$
\mathbf{x}_{t+1} = \begin{bmatrix} x + v_x \\ y + v_y \\ z + v_z \\ \theta \\ l \\ w \\ h \\ v_x \\ v_y \\ v_z \end{bmatrix} =
\begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\      
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\  
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 
\end{bmatrix} \begin{bmatrix}
x \\ y \\ z \\ \theta \\ l \\ w \\ h \\ v_x \\ v_y \\ v_z
\end{bmatrix} = F \mathbf{x}_t
$$
    
The measurement model is even simpler. We use a matrix $$H \in \mathbb{R}^{m \times n}$$ to model the observation of a 3d bounding box  from a 3d object's trajectory state $$\mathbf{x}_t$$. Here $$H$$ is $$7 \times 10$$ in shape:

$$
\mathbf{x}_{t+1} = \begin{bmatrix}
x \\ y \\ z \\ \theta \\ l \\ w \\ h \\ 0 \\ 0 \\ 0
\end{bmatrix} = \begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0
\end{bmatrix} \begin{bmatrix}
x \\ y \\ z \\ \theta \\ l \\ w \\ h \\ v_x \\ v_y \\ v_z
\end{bmatrix} = H \mathbf{x}_t
$$

<a name='3dtracking-icp'></a>
## Practical Example: 3D Tracking with ICP



[Argoverse Tracking Baseline](https://github.com/alliecc/argoverse_baselinetracker)


[1] Xinshuo Weng and Kris Kitani. A Baseline for 3D Multi-Object Tracking. [PDF](https://arxiv.org/pdf/1907.03961.pdf).

[2] Boyd and Vandenberghe. Convex Optimization. [PDF](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)

