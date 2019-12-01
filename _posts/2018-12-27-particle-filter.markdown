---
layout: post
title:  "Particle Filter"
permalink: /particle-filter/
excerpt: "multi-modal distributions, sigma point transform, matrix square roots ..."
mathjax: true
date:   2018-12-27 11:00:00
mathjax: true

---
Table of Contents:
- [Particle Filter Overview](#sfmpipeline)
- [What's Wrong With the EKF?](#costfunctions)
- [A Different Parameterization: Particles](#bundleadjustment)
- [Verify Inverse Unscented Transform](#placeholder)
- [Matrix Square Roots](#placeholder)
- [UKF (Sigma Point Filter)](#placeholder)
- [Choosing Lambda](#placeholder)
- [PRO version of UKF](#placeholder)
- [Predict Step](#placeholder)
- [Sample Degeneracy](#placeholder)


<a name='sfmpipeline'></a>

## Particle Filter Overview

The **Particle Filter** is a filtering algorithm that, unlike the Kalman Filter or EKF, can represent multi-modal distributions. This is because it contains no assumptions about the form of the state distribution. It was published in 1995 [2,3] by Simon Julier, Jeffrey Uhlmann, and Hugh Durrant-Whyte at Oxford. It is often called the **"Unscented Kalman Filter" (UKF)** because the inventors thought "it didn't stink" like the EKF.

The main idea is to Represent a distribution $$p(x)$$ with a collection of samples (particles) from $$p(x)$$: $$x^i \sim p(x), i=1,\dots, N$$ i.i.d. We know that empirical moments are related to true distribution by the Law of Large Numbers. The algorithm is simple, but expensive to compute with a for loop.



## What's Wrong With the EKF?

The Particle Filter addresses a number of problems with the EKF: 

**Problem No. 1** The initial conditions (I.C.s)!
*If your initial guess is wrong, the Kalman filter will tell you exactly the wrong thing. The linearization can be vastly different at different parts of the state space*. For example, the EKF could diverge if the residual $$\| \mu_{0 \mid 0} - x \|$$ on the initial condition is large. In fact, if $$\| \mu_{t \mid t} - x_t \|$$ large at any time, then the EKF could diverge. This has to do with severity of non-linearity. This is the most commonly found problem.

**Problem No. 2: Covariance**
*In the EKF, the "covariance matrix" does not literally represent the covariance.* Unfortunately, the covariance matrix only literally captures the covariance in the Kalman Filter. In the EKF, it is just a matrix! We don't know what it means! If we treat it as confidence, then it is reasonable enough. And commonly true, as long as $\mu$ is tracking $x$ pretty well. However, this estimate tends to be overconfident since we are not incorporating linearization errors! Instead, $$\Sigma_{t \mid t}$$ incorporates only the noise errors $$Q_t, R_t$$. Thus, $$\Sigma_{t \mid t}$$  tends to be smaller than the true covariance matrix.

**Problem No. 3: No Canary in the Goldmine**
An additional problem with the EKF is that we have no signal to know if we're going awry.

## A Different Parameterization: Particles

The UKF represents a different type of compromise than the EKF. In the Kalman Filter and its Extended variant, $$\mu, \Sigma$$ are the parameters that define the distribution $$\mathcal{N}(\mu, \Sigma)$$.

The UKF parameterizes the state distribution in a different way by using "Sigma points." Consequently, the parameterization is called the ($$\sigma$$-points) parameterization. Thus, we move from $$(\mu, \Sigma)$$ to a set of points with a weight associated with each, e.g. $$ \{ (x^0,w^0), \cdots, (x^{2n}, w^{2n}) \}$$.

The "Unscented Transform" is a curve-fitting exercise that converts individual points to a mean and covariance, i.e. $$UT(\mu, \Sigma) = \{ (x^i, w^i) \}_i$$. An advantage of the UKF is that it is very easy to propagate these individual points through nonlinearities like non-linear dynamics, whereas it is harder to push $$\mu, \Sigma$$ through the nonlinearities.


*Properties that we want the unscented transform to have*
\item UT($$\cdot$$):\\
$$(\mu, \Sigma) = \{ (x^i, w^i) \}_i$$

$$UT^{-1}(\cdot)$$ \\
$$ \{ (x^i, w^i) \}_i = (\mu, \Sigma)$$
We want the sample sigma points to share the same mean, covariance

$$
\mu = \sum\limits_{i=0}^{2n} w^ix^i
$$

$$
\Sigma = \sum\limits_{i=0}^{2n} w^i (x^i - \mu)(x^i - \mu)^T
$$

They are redundant (an overparameterization of the Gaussian)

We have redundancy for a smoothing effect, since won't be for a perfect Gaussian

Here is the transform $$UT(\cdot)$$:\\
$$x^0 = \mu$$ \\
$$x^i = \mu + ( \sqrt{(n+\lambda) \Sigma } )_i, i = 1, \dots, n$$\\

This is a matrix square root

The index $$i$$ is the $$i$$'th column in the matrix square root

We also have a mirror image set:\\
$$x^i = \mu - ( \sqrt{(n+\lambda) \Sigma } )_{i-n}, i =n+ 1, \dots, 2n$$\\

Those were the points. The weights themselves:

$$
w^0 = \frac{\lambda}{n+\lambda}
$$

$$
w^i = \frac{1}{2(n+\lambda)}, i \geq 1
$$

Each of these points plot points around the circle/ellipse
Break ellipse into major and minor axes. $$x_1, x_2, x_3, x_4$$ at the corners of the principal axes and the parameters.

$$n \times n$$ matrix is $$\sqrt{(n+\lambda) \Sigma}$$

## Verify Inverse Unscented Transform**

We verify $$UT^{-1}(\cdot)$$ as claimed

$$
\sum\limits_{i=0}^{2n} w^ix^i = \frac{\lambda}{n+\lambda}(\mu) + \sum\limits_{i=1}^n  \frac{1}{2(n+\lambda)} \Bigg(\mu + ( \sqrt{(n+\lambda) \Sigma } )_i \Bigg) + \sum\limits_{i=n+1}^{2n} \frac{1}{2(n+\lambda)} \Bigg(\mu +  - ( \sqrt{(n+\lambda) \Sigma })_{i-n} \Bigg)
$$

$$
= \frac{\lambda}{n+\lambda}(\mu) +  \frac{n}{2(n+\lambda)} \Bigg(\mu  \Bigg) +  \frac{n}{2(n+\lambda)} \Bigg(\mu \Bigg) - \mu
$$

Everything also cancels in the $\Sigma$ calculation


$$
\sum\limits_{i=1}^n \frac{ (\sqrt{n_\lambda})(\sqrt{n_\lambda})}{ (n+\lambda)} (\sqrt{\Sigma}_i) (\sqrt{\Sigma})_i^T = \Sigma
$$

We notice that if $AA^T = B$, then we can decompose $A$ as

$$
A = \begin{bmatrix}
| & \cdots & | \\
a_1 \cdots a_n \\
| & \cdots & | 
\end{bmatrix}
$$

then $$AA^T$$ is


$$
 = \begin{bmatrix}
| & \cdots & | \\
a_1 & \cdots  & a_n \\
| & \cdots & | 
\end{bmatrix}
\begin{bmatrix}
-- & a_1^T & -- \\
-- & \cdots & -- \\
-- & a_n^T & -- | 
\end{bmatrix}
$$

Inverting the transform is just as simple
We get $$a_1 a_1^T + a_2 a_2^T + \cdots + a_n a_n^T$$


$$
\sum\limits_{i=1}^n a_i a_i^T = AA^T = B
$$

therefore

$$
\sum\limits_{i=1}^n (\sqrt{\Sigma})_i (\sqrt{\Sigma})_i^T = (\sqrt{\Sigma}) (\sqrt{\Sigma})^T = \Sigma
$$


## Matrix Square Roots

You might prefer the Cholesky Factorization for numerical versions
Matrix Square Root! Can use SVD or Cholesky Factorization
There are SVD matrix square roots:
(i) $$\sqrt{\Sigma} = U \Lambda^{1/2}$$

$$
\sqrt{\Sigma} (\sqrt{\Sigma})^T = U \Lambda^{1/2} (U \Lambda^{1/2})^T = \Sigma
$$

Columns of $U$ matrix are principal directions of ellipse. SVD gives you this geometric intution of the points around the semi axes, etc.

(ii)

$$
\sqrt{\Sigma} = U \Lambda^{1/2} U^T = \Sigma
$$

SVD computation takes $O(4n^3)$ time

**Cholesky Decomposition**
$$M = LU$$, where lower triangular times upper triangular
When $$M=M^T$$, $$L = U^T$$
Let $$M=LL^T$$
(iii) $$L = \sqrt{\Sigma}$$
People prefer cholesky in UKF because it has complexity $$O( \frac{1}{6} n^3)$$, just a constant savings.

We choose $$\sqrt{\Sigma} \sqrt{\Sigma}^T = LL^T = \Sigma$$

Zero out successive row elements of your vector, it is not along semi axes, but more along different axis-aligned directions as you zero  out rows


## UKF (Sigma Point Filter)


$$
(\mu, \Sigma) \rightarrow UT(\cdot) \rightarrow (x^i, w^i) \rightarrow predict \rightarrow (\bar{x}^i, \bar{w}^i) \rightarrow UT^{-1}(\cdot) \rightarrow  (\bar{\mu}, \bar{\Sigma}) \rightarrow UT(\cdot) \rightarrow (x^i, w^i) \rightarrow update \rightarrow 
$$

**Predict Step**

$$ UT(\cdot) $$

$$
\begin{aligned}
x_{t \mid t}^0 = \mu_{t \mid t} \\
x_{t \mid t}^i = \mu_{t \mid t} + (\sqrt{(n+\lambda) \Sigma_{t \mid t}})_i, i = 1, \dots, n \\
x_{t \mid t}^i = \mu_{t \mid t} - (\cdots), i = n+1, \dots, 2n \\
\bar{x}_{t +1 \mid t}^i = f(x_{t \mid t}^i, u_t)
\end{aligned}
$$

We predict through nonlinear dynamics

Now we run $$UT^{-1}$$

$$
\begin{aligned}
\mu_{t+1 \mid t} = \sum\limits_{i=0}^{2n} w^i \bar{x}_{t+1 \mid t}^i \\
\Sigma_{t+1 \mid t} = \sum\limits_{i=0}^{2n} w^i (\bar{x}_{t+1 \mid t}^i - \mu_{t+1 \mid t})(\bar{x}_{t+1 \mid t}^i - \mu_{t+1 \mid t})^T
\end{aligned}
$$

We recall the Gaussian estimate!

$$
\begin{aligned}
\mu_{t \mid t} = \mu + \Sigma_{XY} \Sigma_{YY}^{-1} (y - \hat{y}) \\
\Sigma_{t \mid t} = \Sigma - \Sigma_{XY} \Sigma_{YY}^{-1} \Sigma_{YX}
\end{aligned}
$$


\item Now the UPDATE step:\\
We run $$UT(\cdot)$$\\
$$x_{t +1 \mid t}^0$$\\
$$x_{t+1 \mid t}^{2n}$$\\

Let's build $\hat{y}_{t+1 \mid t}$ and $\Sigma_{t+1 \mid t}^{XY}$, $\Sigma_{t+1 \mid t}^{YY}$
\item Now

$$
\hat{y}_{t+1 \mid t} = \sum\limits_{i=0}^{2n} w^i \hat{y}_{t+1 \mid t}^i
$$

which is the expected measurment
Now,


$$
\begin{aligned}
\Sigma_{t+1 \mid t}^{YY} = \sum\limits_{i=0}^{2n} w^i (\hat{y}_{t+1 \mid t}^i - \hat{y}_{t+1 \mid t}) (\hat{y}_{t+1 \mid t}^i - \hat{y}_{t+1 \mid t})^T
\end{aligned}
$$

Now

$$
\begin{aligned}
\Sigma_{t+1 \mid t}^{XY} = \sum\limits_{i=0}^{2n} w^i (x_{t+1 \mid t}^i - \mu_{t+1 \mid t}) (\hat{y}_{t+1 \mid t}^i - \hat{y}_{t+1 \mid t})^T
\end{aligned}
$$

We are doing a fitting operation. That is why we have more sigma points than we need. Smooth out the anomalies due to any one point getting weird.

\begin{equation}
\Sigma_{t+1 \mid t+1} = \Sigma_{t+1 \mid t} - \Sigma_{t+1 \mid t}^{XY} \Sigma_{t+1 \mid t}^{YY}
\end{equation}





\begin{equation}
\mu_{t+1 \mid t+1 } = \mu_{t+1 \mid t } + \Sigma_{t+1 \mid t}^{XY} (\Sigma_{t+1 \mid t}^{YY})^{-1} (y_{t+1} - \hat{y}_{t+1 \mid t})
\end{equation}

where $$y_{t+1}$$ is the actual measurement.

## Choosing Lambda

 What is $$\lambda$$? Problem specific.  Consider SVD square root, so that sigma points will be along principal axes.

Suppose we have $$x^0$$ at the center of the ellipse, and $$x^1, \dots, x^4$$ lie at each corner of the principal semi-axes

$$\lambda$ is the "confidence-value" of the error ellipse

If $$n+\lambda = 1$$, then $$x^i = \mu \pm \sqrt{\Sigma}_i$$ and each column represents one standard deviation

Standard deviation in each direction. Bigger $$\lambda$$ is, then the bigger is the ellipse (And vice versa: smaller $$\lambda$$ gives smaller ellipse)

The size of the ellipse matters because this is what we take as the region about which we create our linearization

UKF is a linearization, takes average slope over a neighborhood. But it is not from the Taylor Series Expansion

Why does size of ellipse matter? Blurring over a bigger neighborhood. (neighborhood about which we fit the Gaussian is determined by  $$\lambda$$

The smaller $$\lambda$$ is, the closer it will be to an EKF, which fits about a single-point (linearizing it there)

Interesting value of $$\lambda$$: $$\lambda=2$$. For a quadratic non-linearity, then the inverse unscented transform fits the mean and the covariance of the Gaussian, and also the Kurtosis (the 4th moment) of the Gaussian (but only for a quadratic nonlinearity)

Fitting the Kurtosis is good! We can do it beacause the extra degrees of freedom of the sigma points overparameterize

## PRO version of UKF


Other Form of UKF: 

\begin{equation}
\lambda = \alpha^2 ( n + k) - n
\end{equation}
this gives us two parameters to tune

$$
x^i = \mu \pm \alpha ( \sqrt{(n+k) \Sigma})_i
$$

where the two parameters are $$\alpha,k$$

We now have to redefine the weights to be

$$
w_c^0 = \frac{\lambda}{n+\lambda} + (1 - \alpha^2 + \beta)
$$

where $$\beta$$ is another parameter                       

Hugh Durrant White, the original paper has this original form

How does it work?


 Algorithm is simple, but expensive to compute (with for loop)

In UKF, samples deterministally extracted

In Particle filter, no assumption about form of distribution, but need many form of them, and probabilistically extracted

**Main Idea:** Represent a distribution $p(x)$ with a collection of samples from $$p(x)$$: $$x^i \sim p(x), i=1,\dots, N$$ i.i.d.

What does it mean to ``represent'' a distribution $$p(x)$$ with a bunch of particles $$\{ x_1, \dots, x_N \}$$?

We know that empirical moments are related to true distribution by the law of large number

Recall

$$
\begin{array}{lll}
\mu = \mathbb{E}[X] = \int_x p(x) dx \approx \frac{1}{N} \sum\limits_{i=1}^N x_i = \bar{\mu}, & x_i \sim p(x), & i=1,\dots, N, i.i.d
\end{array}
$$

Then 

$$
\begin{array}{ll}
\Sigma = \mathbb{E}[(X-\mu)(X-\mu)^T] = \dots \approx \frac{1}{N} \sum\limits_{i=1}^N (x_i - \bar{\mu})(x_i - \bar{\mu})^T, & x_i \sim p(x)
\end{array}
$$

Also,

$$
\mathbb{E}[f(x)] \approx \frac{1}{N} \sum\limits_{i=1}^N f(x_i)
$$

The Law of Large numbers states that

$$
\frac{1}{N} \sum\limits_{i=1}^N f(x_i) \rightarrow \mathbb{E}[f(X)]
$$

as $$N \rightarrow \infty$$

Problem: Given a set of particles 

$$
\{ x_{t+1 \mid t}^1, \dots, x_{t+1  \mid t}^N \} \sim p(x_{t+1} \mid y_{1:t} )
$$

drawn from the above distribution

We recall the update step

$$
p(x_t \mid y_{1:t} ) = \frac{ p(y_t \mid x_t) p(x_t \mid y_{1:t-1}) }{ \int_{x_t} p(y_t \mid x_t) p(x_t \mid y_{1:t-1}) dx_t }
$$

We have particles from the top right expression, $$x_{t \mid t-1}^i \sim p( x_t \mid y_{1:t-1})$$

We want to use Bayes Rule so that we can transform our particles so that they approximate the posterior (this will be one step of the Bayesian Filter)

We will use :


$$q(x)$$ the proposal distribution, the particles are actually from here

We want them to be from $$p(x)$$ the target distribution, we wish they were from here

We use Particle Weights

$$
\mathbb{E}_p [ f(X)] = \int_x f(x) p(x) dx = \int_x f(x) p(x) \frac{q(x)}{q(x)} dx 
$$


$$
\mathbb{E}_p [ f(X)] = \int_x f(x) w(x) q(x) dx  = \mathbb{E}_q [ f(X) w(x) ]
$$

Given $$\{x^1, \dots, x^N \}$$
New set: $$\{ (x^1,w^1), \dots, (x^N,w^N) \}$ where $w^i = w(x^i)$$
Now the expectation is

$$
\mathbb{E}_p[f(X)] \approx \frac{1}{N} \sum\limits_{i=1}^N f(x^i) w^i
$$

and $$w(x) = \frac{p(x)}{q(x)}$$

If we knew $$p,q$$, then we would know the weights
But in the filtering setup, we don't know the posterior, the distribution we are trying to hit! But in fact, we don't need the target distribution!

Proposal: What we have

$$
q(x) = p(x_t \mid y_{1:t-1})
$$

Target: What we want

$$
p(x) = p(x_t \mid y_{1:t})
$$

So we get weights:

$$
w^i = w(x^i) = \frac{ p(x^i) }{ q(x^i) } = \frac{p(x_t^i \mid y_{1:t})^i}{p(x_t^i \mid y_{1:t-1}) }
$$

Now, by Bayes Rule, we can say that the PRIOR cancels out, which was the only thing we didn't know???

$$
 = \frac{     \frac{p(y_t \mid x_t) p(x_t \mid y_{1:t-1}) }{ \int_{x_t} \cdots dx_t }     }{ p(x_t \mid y_{1:t-1}) }
$$

When the prior cancels, we get

\begin{equation}
w(x_t^i) = \frac{p(y_t \mid x_t^i ) }{ \int_{x_t} \cdots dx_t } 
\end{equation}

We just care about the relative weights, so

$$
\bar{w} (x_t^i) = p(y_t \mid x_t^i)
$$

(unnormalized)

$$
w(x_t^i) = \frac{ \bar{w}(x_t^i) }{  \sum\limits_{i=1}^N \bar{w}(x_t^i) }
$$

Example: suppose we have measurement noise $v_t \sim \mathcal{N}(0, R_t)$

$$
y_t = g(x_t) + v_t
$$

Given $x_{t \mid t-1}^i \sim p(x_t \mid y_{1:t-1})$, then find

$$
\{ (x_{t \mid t}^i, w_{t \mid t}^i \}_i
$$

to represent $p(x_t \mid y_{1:t})$. The weights are

$$
\bar{w}_{t \mid t}^i = p(y_t \mid x_t = x_{t \mid t-1}^i) \sim \mathcal{N}( g(x_{t \mid t-1}^i, R_t)
$$

And now

$$
\bar{w}_{t \mid t}^i = \eta \mbox{exp } \{ -\frac{1}{2} (y_t - g(x_{t \mid t-1}^i)^T R_t^{-1} (y_t - g(x_{t \mid t-1}^i) ) \}
$$

Renormalize

$$
w(x_t^i) = \frac{ \bar{w}(x_t^i) }{ \sum\limits_{i=1}^N \bar{w}(x_t^i)  }
$$

Then multiply the weights, and renormalize (could have started with weighted particles instead of just particles)

Only the weight changes in the UPDATE step (but we keep two different time indices for weights). Time indices are redundant for the $$x$$ particles, which are the input

$$
\{ (x_t^i, w_{t \mid t-1}^i) \}
$$

$$
\bar{w}_{t \mid t}^i = p(y_t \mid x_t = x_t^i) w_{t \mid t-1}^i
$$

and

$$
w_{t \mid t}^i  = \frac{ \bar{w}_{t \mid t}^i }{ \sum\limits_{i=1}^N \bar{w}_{t \mid t}^i  }
$$

## Predict Step

$$
p(x_{t+1} \mid y_{1:t}) = \int_x p(x_{t+1} \mid x_t) p(x_t \mid y_{1:t}) dx_t
$$

the left side is the transition distribution (which we have)

the right side is $\{ (x_t^i, w_{t \mid t}^i) \}$

Let's just "simulate" $$p(x_{t+1} \mid x_t)$$

Draw $$x_{t+1}^i \sim p(x_{t+1} \mid x_t = x_t^i )$$

where $$x_{t+1} = f(x_t, u_t) + w_t$$, the process noise $$w_t \sim \mathcal{N}(0, Q_t)$$

We sample 
$$
\begin{array}{ll}
w_t \sim \mathcal{N}(0, Q_t), & \rightarrow x_{t+1}^i = f(x_{t}^i, u_t) + w_t 
\end{array}
$$

1000 particles per dimension $$(1000^{D}$$ where $$D$$=dimension

Exponential explosion of particles required to keep the same resolution


Full Filter: Given

$$
\{ (x_{t }^i, w_{t \mid t}^i \}_i
$$

Predict:

$$
x_{t+1}^i \sim p(x_{t+1} \mid x_t = x_t^i )
$$

Update:

$$
\bar{w}_{t+1 \mid t+1}^i = p(y_{t+1} \mid x_{t+1} = x_{t+1}^i) w_{t \mid t}^i
$$

$$
w_{t+1 \mid t+1 }^i  = \frac{ \bar{w}_{t+1 \mid t+1}^i }{ \sum\limits_{i=1}^N \bar{w}_{t+1 \mid t+1}^i  }
$$

## Sample Degeneracy

Unfortunately, it is easy to end up with degenerate samples in the UKF, meaning the samples won't do anything for you because they are all spread apart. TODO: Only one takes the weight, why?

The **solution**: resample points in a very clever way, via importance (re-)sampling. In a *survival of the fittest*, those samples with high weight end up lots of children, and those with low weight disappear. I


\item We sample a new set of particles at time $t$, get new $x_t^i$ s.t.
\item This is sequential important re-sampling.

$$
Pr( x_t^i = x_t^j) = w_t^j
$$

The noisy prediction will disperse them. The new weights are uniform, i.e. $$w_t^i = \frac{1}{N}$$
\item Particle impoverishment; all particles clumped at one point because not enough noise to disperse them
\item If the resampling happens too frequently, in comparison with the dispersal via noise, then we get sample impoverishment!

We will get that all $x_t^i \approx x_t^j$, and $w_t^i \approx \frac{1}{N}, \forall i$

So no diversity in the distribution anymore

To fix it, resample less often

 Or you could just throw some random particles into your bag

Check how different weights are at every time. If too different, trigger a resample (variance without subtracting mean is $$\sum\limits_{i=1}^N (w_t^i)^2 \geq \mbox{thresh}$$

If maximally different, then you would get $$1$$ and $$0$$, maximally different



**Low Variance Resampling:** in previous formulation, you could have gotten all copies of the same particle
One number line is $$r \sim v[0, \frac{1}{N}]$$
Then map each of these to the right interval above in the weighted bins. Determinsitic resampling
Make $$n$$ copies of it in every uniform bin

**Swept under Rug**
All filters have tried to compute the Bayesian posterior (either exactly or approximately)
The particle filter does not do this (look at prediction step)
PF tracks approximates the posterior of the trajectory instead

$$
p(x_{0:t} \mid y_{1:t})
$$

Because of the prediction step, where we say draw sample

$$
x_{t+1}^i \sim p(x_{t+1} \mid x_t  = x_t^i )
$$

What we really wanted was

$$
x_{t+1}^i \sim p(x_{t+1} \mid y_{1:t} ) = \int_{x_t} p(x_{t+1} \mid x_t) p(x_t \mid y_{1:t}) dx_t
$$

\item Instead, the Particle Filter finds

$$
x_{t+1}^i \sim p(x_{t+1}, x_t \mid y_{1:t}) = p(x_{t+1} \mid x_t = x_t^i) p(x_t = x_t^i \mid y_{1:t})
$$

Track distribution of the whole TRAJECTORY, given all of the measurements.
Important for SLAM, because in SLAM we often want to estimate the history of the trajectory of the robot
\end{itemize}


## References

[1] Mac Schwager. Lecture Presentations of AA 273: State Estimation and Filtering for Aerospace Systems, taught at Stanford University in April-June 2018. 

[2] SJ Julier, JK Uhlmann, HF Durrant-Whyte. *A new approach for filtering nonlinear systems*. Proceedings of the American Control Conference, June 1995, Volume 3, pages 1628-1632.

[3] Simon Julier, Jeffrey Uhlmann, and Hugh F. Durrant-Whyte. *A New Method for the Nonlinear Transformation of Means and Covariances in Filters and Estimators*. IEEE Transactions on Automatic Control, Volume 45, No. 3. March 2000, page 477.






