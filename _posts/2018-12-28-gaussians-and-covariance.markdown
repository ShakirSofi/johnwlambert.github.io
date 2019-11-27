---
layout: post
title:  "Understanding Multivariate Gaussians and Covariance"
permalink: /gauss-covariance/
excerpt: "error ellipses, uncertainty "
mathjax: true
date:   2018-12-27 11:00:00
mathjax: true

---
Table of Contents:
- [A Basic SfM Pipeline](#sfmpipeline)
- [Cost Functions](#costfunctions)
- [Bundle Adjustment](#bundleadjustment)

<a name='sfmpipeline'></a>

## What is a multivariate Gaussian random variable?

Gaussian R.V.s are parameterized by two quantities:

$$
X \sim p(x)  = \mathcal{N}(\mu_x, \Sigma_x)
$$

Preceded by a term for normalization

$$
\mathcal{N}(\mu_x, \Sigma_x) = 
\frac{1}{\sqrt{(2\pi)^n|\Sigma_x|}}\mbox{exp}
\Bigg\{ -\frac{1}{2} (x - \mu_x)^T \Sigma_x^{-1} (x - \mu_x) \Bigg\}
$$

where $$n$$ is the dimension, i.e. $$X \in \mathbb{R}^n$$

And of course in the scalar case, we see

$$
\mathcal{N}(\mu_x, \sigma_x) = 
$$

Level sets trace out ellipses that are centered at $$\mu_x$$, have minor and major axes (in 2D), and ellipsoids in higher dimensions

Mean:

$$
E[X] = \int_x x p(x) dx
$$

need to show

$$
\mu_x = \int_x x \mathcal{N}(\mu_x,\Sigma_x) dx
$$

 Covariance

$$
E[(X- \mu_x)(X - \mu_x)^T] = \int_x (x-\mu_x)(x-\mu_x)^T p(x) dx
$$

need to show

$$
\Sigma_x = \int_x (x- \mu_x) (x-\mu_x)^T \mathcal{N}(\mu_x, \Sigma_x) dx
$$

Gaussian distribution is a second-order distribution, which does not mean that the higher order moments are zero (not true in general)
Convert between a standard normal, and any multivariate Gaussian

## The Standard Normal Distribution

$$
X_s \sim \mathcal{N}(0, I)
$$

unit (identity) covariance, so each axis decouples, compute integral over each axis separately

**How to generate any Gaussian R.V. from standard normal** $$X_s$$?

$$
X = \Sigma_x^{1/2}X_s + \mu_x
$$

Can obtain by scaling by covariance matrix, and by translating by mean.
Very good for simulating. With MATLAB or Python, can generate scalar, unit variance, 0 mean Gaussian R.V. with `numpy.random.randn`. Call `numpy.random.randn` $$n$$ times to populate $$X_s$$, and them multiply, then add. We can compute such a matrix square root via the Cholesky Decomposition (which will be unique if $$\Sigma_x$$ is positive definite):

$$
\Sigma_x = \Sigma_x^{1/2}(\Sigma_x^{1/2})^T
$$

## Matrix Square Roots

There are other possible matrix square roots
**How to transform any Gaussian R.V. to the standard normal** $$X_s$$?
We can do so via rearrangement of the expression exactly above:

$$
\begin{array}{ll}
X_s = \Sigma_x^{-1/2}(X - \mu_x), & X \sim \mathcal{N}(\mu_x, \Sigma_x)
\end{array}
$$

Much easier to integrate over the form on the RHS, not LHS
Comes from method of derived distributions
Derived Distributions: Given $$X \sim p(x)$$, $$Y=f(X)$$, find $$p(y)$$
Here $$X = X_s$$, and function $$f$$ is the linear distribution $$AX + b$$.




## Properties of Multivariate Gaussians


## How can we understand a covariance matrix?

Larger covariance means more uncertainty. Isocontours/error ellipses

We set $$P=0.95$$:

$$
\varepsilon = \frac{1-P}{2 \pi |\Sigma|^{1/2}} = \frac{1-0.95}{2 \pi |\Sigma|^{1/2}} = \frac{0.05}{2 \pi |\Sigma|^{1/2}}
$$


$$
 \mathcal{B}(\varepsilon) = \Big\{ x \mid p(x) \geq \varepsilon \Big\}
$$


## Covariance

Cavariance Matrix How can we determine the direction of maximal variance? The first
we can do is to determine the variances of the individual components. If the data points (or
vectors) are written as x = (x1, x2)T (T indicates transpose), then the variances of the first
and second component can be written as C11 := "x1x1# and C22 := "x2x2# (angle brackets
indicate averaging over all data points). If C11 is large compared to C22, then the direction of
maximal variance is close to (1, 0)T , while if C11 is small, the direction of maximal variance
is close to (0, 1)T . (Notice that variance doesn’t have a polarity, so that one could use the
inverse vector (−1, 0)T instead of (1, 0)T equally well for indicating the direction of maximal
variance.)
But what if C11 is of similar value as C22, like in the example of Figure 1? Then the
co-variance between the two components, C12 := "x1x2#, can give us additional information
(notice that C21 := "x2x1# is equal to C12). A large positive value of C12 indicates a strong
correlation between x1 and x2 and that the data cloud is extended along the (1, 1)T direction.
A negative value would indicate anti-correlation and an extension along the (−1, 1)T
direction. A small value of C12 would indicate no correlation and thus little structure of
the data, i.e. no prominent direction of maximal variance. The variances and covariances
are conveniently arranged in a matrix with components Cij , which is called covariance matrix
(assuming zero mean data). Figure 3 shows several data clouds and the corresponding
covariance matrices.

0.2 0
0 1


1 -0.5
-0.5 0.3


1 0 
0 1




```

import numpy as np
import pdb
import matplotlib.pyplot as plt
import seaborn as sns


import scipy

sns.set_style({'font.family': 'Times New Roman'})

def plot_gauss_ellipse(mu, cov, color='g', rad=2, ):
	"""
	Adapted from Piotr Dollar's https://github.com/pdollar/toolbox/
	Plots a 2D ellipse derived from a 2D Gaussian specified by mu & cov.
	
	USAGE:
		hs = plotGaussEllipses( mus, Cs, [rad] )
	
	Args:
	-	mus: Numpy array of shape (2,), representing mean
	-	Cs: Numpy array of shape (2,2), representing covariance matrix
	-	color: string representing Matplotlib color
	-	rad: [2] Number of std to create the ellipse to
	
	Returns:
	-	None
	
	color choices: ['b', 'g', 'r', 'c', 'm', 'y', 'k']
	"""
	cRow, ccol, ra, rb, phi = gauss2ellipse( mu, cov, rad)
	plotEllipse( cRow, ccol, ra, rb, phi, color)



def gauss2ellipse(mu, C, rad=2):
	"""
	Adapted from Piotr Dollar's https://github.com/pdollar/toolbox/
	Creates an ellipse representing the 2D Gaussian distribution.
	
	Creates an ellipse representing the 2D Gaussian distribution with mean mu
	and covariance matrix C.  Returns 5 parameters that specify the ellipse.
	
	USAGE
	 [cRow, cCol, ra, rb, phi] = gauss2ellipse( mu, C, [rad] )
	
	Args:
	-	mu: 1x2 vector representing the center of the ellipse
	-	C: 2x2 cov matrix
	-	rad: [2] Number of std to create the ellipse to
	
	OUTPUTS
	-	cRow: the row location of the center of the ellipse
	-	cCol: the column location of the center of the ellipse
	-	ra: semi-major axis length (in pixels) of the ellipse
	-	rb: semi-minor axis length (in pixels) of the ellipse
	-	phi: rotation angle (radians) of semimajor axis from x-axis
	
	EXAMPLE
	#  [cRow, cCol, ra, rb, phi] = gauss2ellipse( [5 5], [1 0; .5 2] )
	#  plotEllipse( cRow, cCol, ra, rb, phi );
	"""
	# error check
	if mu.size != 2 or C.shape != (2,2):
		print('Works only for 2D Gaussians')
		quit()

	# decompose using SVD
	_,D,Rh = np.linalg.svd(C)
	R = Rh.T
	normstd = np.sqrt(D)

	# get angle of rotation (in row/column format)
	phi = np.arccos(R[0,0])

	if R[1,0] < 0:
		phi = 2*np.pi - phi
	phi = np.pi/2 - phi

	# get ellipse radii
	ra = rad * normstd[0]
	rb = rad * normstd[1]

	# center of ellipse
	cRow = mu[0]
	cCol = mu[1]

	return cRow, cCol, ra, rb, phi



def plotEllipse(cRow,cCol,ra,rb,phi,color='b',nPnts=100,lw=1,ls='-'):
	"""
	Adapted from Piotr Dollar's https://github.com/pdollar/toolbox/
	Adds an ellipse to the current plot.
	
	USAGE:
	-	h,hc,hl = plotEllipse(cRow,cCol,ra,rb,phi,[color],[nPnts],[lw],[ls])
	
	Args:
	-	cRow: the row location of the center of the ellipse
	-	cCol: the column location of the center of the ellipse
	-	ra: semi-major axis radius length (in pixels) of the ellipse
	-	rb: semi-minor axis radius length (in pixels) of the ellipse
	-	phi: rotation angle (radians) of semimajor axis from x-axis
	-	color: ['b'] color for ellipse
	-	nPnts: [100] number of points used to draw each ellipse
	-	lw: [1] line width
	-	ls: ['-'] line style

	Returns:
	-	h : handle to ellipse
	-	hc: handle to ellipse center
	-	hl: handle to ellipse orient

	EXAMPLE:
		plotEllipse( 3, 2, 1, 5, pi/6, 'g');
	"""
	# plot ellipse (rotate a scaled circle):
	ts = np.linspace(-np.pi, np.pi, nPnts+1)
	cts = np.cos(ts)
	sts = np.sin(ts)

	x = ra * cts * np.cos(-phi) + rb * sts * np.sin(-phi) + cCol
	y = rb * sts * np.cos(-phi) - ra * cts * np.sin(-phi) + cRow
	h = plt.plot(x,y, color=color, linewidth=lw, linestyle=ls)

	# plot center point and line indicating orientation
	hc = plt.plot(cCol, cRow, 'k+', color=color, linewidth=lw, linestyle=ls)

	x = [cCol, cCol+np.cos(-phi)*ra]
	y = [cRow, cRow-np.sin(-phi)*ra]
	hl = plt.plot(x, y, color=color, linewidth=lw, linestyle=ls)

	return h,hc,hl


def gen_from_distribution():



	Sigma_sqrt = scipy.linalg.sqrtm(Sigma)
	Sigma_inv = np.linalg.inv(Sigma)
	tiled_mu = np.tile(mu,(1000,1)).T
	samples = np.matmul( Sigma_sqrt, np.random.randn(2,1000) ) + tiled_mu
	exterior_samples = np.zeros((1,2))
	interior_samples = np.zeros((1,2))

	for i in range(samples.shape[1]):
			X = samples[:,i]
			f_val = 0.5 * np.matmul( np.matmul( (X - mu).T, Sigma_inv), X - mu )
			if f_val < -np.log(0.05):
				interior_samples = np.vstack([interior_samples, np.reshape(X,(1,2)) ])
			else:
				exterior_samples = np.vstack([exterior_samples, np.reshape(X,(1,2)) ])

	plt.scatter(interior_samples[:,0], interior_samples[:,1], c= 'b')
	plt.scatter(exterior_samples[:,0], exterior_samples[:,1], c= 'b')

def plot_gauss_ellipse_v2():
	"""
	"""
	d = 2 # dimension of samples
	p = 0.95 # probability
	num_samples = 1000

	mu = np.array(0,0) # dimension (2,)

	# define covar matrices of dim (2,2)
	cov_mats[0] = np.array([[1,0],
							[0,1]])
	cov_mats[1] = np.array([[2,0],
							[0,2]])
	cov_mats[2] = np.array([[0.25, 0.3]
							[0.3, 1]])
	cov_mats[3] = np.array([[10., 5]
							[5., 5]])

	for covar_mat in cov_mats:


		# generate and plot 1000 samples
		generate_gaussian_samples()
		plt.plot()

		# plot the error ellipse
		r = np.sqrt(ellipse_const)
		n_pts = int((2*np.pi) / 0.01)+1
		theta = np.linspace(0,2*np.pi,n_pts)
		w1 = r * np.cos(theta)
		w2 = r * np.sin(theta)
		w = np.array([w1,w2]).reshape(2,1)

		# transferred back to x coordinates
		x = scipy.linalg.sqrtm(sigma).dot(w) + mu
		plt.plot()


def unit_test1():
	"""
	"""

	# plot_gauss_ellipse(mu=np.array([10., 10.]), cov=np.eye(2))
	# plot_gauss_ellipse(mu=np.array([10., 10.]), cov=np.eye(2)*2.)
	# plt.show()

	fig()

	plot_gauss_ellipse(mu=np.array([10., 10.]), cov=np.array([[5,0],[0,3]]) )
	plt.show()




if __name__ == '__main__':
	unit_test1()
```


