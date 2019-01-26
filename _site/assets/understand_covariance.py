
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


