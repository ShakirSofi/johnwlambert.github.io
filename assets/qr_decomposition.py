

import numpy as np
import pdb

# def darve_house(x):
#     """Computes the Householder transformation for input vector x"""
#     sigma = x[2:].dot(x[2:])
#     v = copy(x)
#     v[1] = 1
#     if (sigma == 0) and (x[1] >= 0):
# 		beta = 0
#     elif (sigma == 0) and (x[1] < 0):
# 		beta = -2
#     else:
# 		mu = np.sqrt(x[1]*x[1] + sigma)
# 		if x[1] <= 0:
# 			v[1] = x[1] - mu
# 		else:
# 			v[1] = -sigma / (x[1] + mu)

# 		beta = 2 * v[1] * v[1] / (sigma + v[1]*v[1])
# 		v /= v[1]

#     return (v, beta)



# def darve_QR_householder_demo():

# 	# Initialize matrix
# 	m = n = 32
# 	Z = np.zeros( (m,n) )
# 	for i in range(m):
# 		for j in range(n):
# 			A[i,j] = np.cos( (np.pi*i*j) / n)

# 	# Apply sequence of Householder transformations
# 	for k in range(n-1):
# 		v,beta = house(A[k:,k])
# 		A[k:,k:] = A[k:,k:] - beta * np.outer(v, v.dot(A[k:,k:]))
# 		# record a snapshot of A



def householder(A):
	""" 
	What if top left entry of A was zero anyways?
	"""
	n, _ = A.shape
	x = A[:,0].reshape(n,1)
	e1 = np.zeros((n,1))
	e1[0] = 1.
	v = np.sign(x[0]) * np.linalg.norm(x) * e1 + x
	# divide by 2-norm squared
	P = np.eye(n) - 2 * v.dot(v.T)/v.T.dot(v)
	return P





def qr_decomposition(A):
	""" 
	Form the QR decomposition of A by using
	Householder matrices.
	"""
	n, _ = A.shape
	P = np.zeros((n-1,n,n))
	R = A
	for i in range(n-1):

		P[i] = np.eye(n)
		# modify lower-right block
		P[i,i:,i:] = householder(R[i:,i:])
		print(f'On iter {i}', np.round(P[i].dot(R), 1))
		R = P[i].dot(R)

	Q_T = np.eye(n)
	for i in range(n-1):
		Q_T = P[i].dot(Q_T)

	Q = Q_T.T
	return Q, R


def qr_demo():
	""" """
	# for i in range(100):
	n = 4
	A = np.random.randint(-100,100,(n,n))

	# print('Numpy QR: ')
	Q_np, R_np = np.linalg.qr(A)
	# print('Numpy Q= ', Q_np)
	# print('Numpy R= ', R_np)
	# print('Numpy QR=', np.round(Q_np.dot(R_np),1))

	print('A=', A)
	Q,R = qr_decomposition(A)

	
	print('Q=',Q)
	print('R=',np.round(R,1))
	QR = Q.dot(R)
	print('QR=', np.round(QR,2))
	print('A=', A)
	print('A-QR=', (A - QR))
	print('Sum: ', np.absolute(A - QR).sum())
	assert( (A - QR).sum() < 1e-10 )

if __name__ == '__main__':
	qr_demo()


