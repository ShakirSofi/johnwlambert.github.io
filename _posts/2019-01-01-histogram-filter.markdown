---
layout: post
title:  "The Histogram Filter"
permalink: /histogram-filter/
excerpt: " "
mathjax: true
date:   2018-12-27 11:00:00
mathjax: true

---
Table of Contents:
- [Need for Dimensionality Reduction](#need-for-dr)
- [PCA](#pca)


<a name='need-for-dr'></a>

## The Histogram Filter

An alternative to continuous distributions are piecewise constant approximations, such as histograms. These are *nonparametric* filters, since they do not utilize parameters like a mean and covariance $$(\mu, \Sigma)$$ to define a distribution. Thus, nonparametric filters do not rely on a *fixed functional form of the posterior*, like the Gaussian does [1].

The Histogram Filter is a type of nonparametric filters that discretizes the state space into a finite number of regions. The histogram assigns to each region a single cumulative probability.

## 1-D Histogram Filter

+ for all \\(k\\) do:
	- \\( \hat{p}_{k,t} = \sum\limits_i p(X_t = x_k \mid u_t, X_{t-1} = x_i) p_{i,t-1} \\)
	- \\( p_{k,t} = \eta p(z_t \mid X_t = x_k) \hat{p}_{k,t}) \\)



## 2-D Histogram Filter and Grid Localization


## References

[1] Sebastian Thrun, Wolfram Burgard, Dieter Fox. *Probabilistic Robotics*. The MIT Press, Cambridge, MA, 2005.




