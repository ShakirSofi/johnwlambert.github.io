


import numpy as np
import pdb







# def compute_schur_decomposition():
# 	""" """
# 	dim = 2
# 	max_iters = 100
# 	A = np.array([	[1,2],
# 					[2,3]])

# 	Q = np.zeros((max_iters,dim,dim))
# 	A_ = A.copy()
# 	for i in range(max_iters):
# 		Q_,R = qr_decomposition(A_)
# 		A_ = R.dot(Q_)
# 		Q[i] = Q_

# 	# compute Schur now...
# 	Q_s = np.eye(dim)
# 	for i in range(max_iters):
# 		Q_s = Q_s.dot(Q[i])
	
# 	Lambda = Q_s.T.dot(A).dot(Q_s)

# 	print('Q = ', Q_s)
# 	print('Lambda = ', Lambda)


# def qr_decomposition(A):
# 	""" 
# 	Form the QR decomposition of A by using
# 	Householder matrices.
# 	"""
# 	n, _ = A.shape
# 	P = np.zeros((n-1,n,n))
# 	R = A
# 	for i in range(n-1):

# 		P[i] = np.eye(n)
# 		# modify lower-right block
# 		P[i,i:,i:] = householder(R[i:,i:])
# 		print(f'On iter {i}', np.round(P[i].dot(R), 1))
# 		R = P[i].dot(R)

# 	Q_T = np.eye(n)
# 	for i in range(n-1):
# 		Q_T = P[i].dot(Q_T)

# 	Q = Q_T.T
# 	return Q, R


# def householder(A):
# 	""" 
# 		Args:
# 		-   A: A n-d numpy array of shape (n,n) 

# 		Returns:
# 		-   P: A n-d numpy array of shape (n,n), representing 
# 		a Householder matrix				
# 	"""
# 	n, _ = A.shape
# 	x = A[:,0].reshape(n,1)
# 	e1 = np.zeros((n,1))
# 	e1[0] = 1.
# 	v = np.sign(x[0]) * np.linalg.norm(x) * e1 + x
# 	# divide by 2-norm squared
# 	P = np.eye(n) - 2 * v.dot(v.T)/v.T.dot(v)
# 	return P


def power_method():
	"""
	"""



if __name__ == '__main__':



