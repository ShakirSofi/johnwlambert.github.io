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
- [A Basic SfM Pipeline](#sfmpipeline)
- [Cost Functions](#costfunctions)
- [Bundle Adjustment](#bundleadjustment)

<a name='sfmpipeline'></a>

## What is the Kalman Filter?

The Kalman Filter is nothing more than the Bayes' Filter for Gaussian distributed random variables. The Bayes' Filter is described in a [previous post](/bayes-filter/).

It is a very surprising result that we can write out the integrals analytically for the Bayes' Filter when working with a special family of distributions: Gaussian distributed random variables (r.v.'s).  As we recall from the Bayes' Filter, we have three quantities that we'll need to be able to evaluate:
- (1) $$ \int_{x_{t-1}} p(x_t \mid x_{t-1}) p(x_{t-1} \mid y_{1:t-1}) dx_{t-1} $$
- (2) $$ p(x_t \mid x_{t-1})  = f(x_t, x_{t-1} ) $$
- (3) $$ p(y_t \mid x_t) = g(y_t, x_t) $$

Expressions (2) and (3) must be finitely parameterizable. It will take a fair amount of work to derive the analytical formulas of the Bayes' Filter for Gaussian r.v.'s (the Kalman Filter).  We'll first review properties of multivariate Gaussians, then the Woodbury matrix inversion lemmas, intuition behind covariance matrices, and then derive the Kalman Filter.


## Woodbury Matrix Inversion Lemmas

## Kalman Filter Derivation


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


## Practical Example: Tracking 3D Bounding Boxes with Constant Velocity


[A Baseline for 3D Multi-Object Tracking](https://github.com/xinshuoweng/AB3DMOT)


## Practical Example: 3D Tracking with ICP



[Argoverse Tracking Baseline](https://github.com/alliecc/argoverse_baselinetracker)





