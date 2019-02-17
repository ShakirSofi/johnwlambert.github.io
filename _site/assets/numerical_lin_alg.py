
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pdb



# def record_snapshot(ims, A):
# 	ims += [A]



# def QR_householder_demo():

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


# def gram_schmidt_demo():
# 	""" """
# 	m = n = 64
# 	A = np.zeros( (m,n) )
# 	for i in range(m):
# 		for j in range(n):
# 			A[i,j] = np.cos( (np.pi*i*j) / n)

# 	B = A.copy()

# 	# Apply the Modified Gram-Schmidt orthogonalization process
# 	for k in range(n):
# 		# Make column k orthogonal to all previous columns
# 		for r in range(k):
# 			A[:,k] = A[:,k] - A[:,r].dot(A[:,k]) * A[:,r]
# 		# Normalize column
# 		A[:,k] = A[:,k] / np.linalg.norm( A[:,k] )





# def LU_column_up_looking(A,ims):
# 	""" """
# 	for j in range(m):
# 		# Update above the diagonal
# 		for i in range(1,j+1):
# 			# Dot product
# 			for k in range(i):
# 				A[i,j] = A[i,j] - A[i,k] * A[k,j]
# 			# record snapshot of A

# 		# Update below the diagonal
# 		for i in range(j+1,m):
# 			# Dot product
# 			for k in range(j):
# 				A[i,j] = A[i,j] - A[i,k] * A[k,j]
# 	pdb.set_trace()



# def LU_column_down_looking(A,ims):
# 	""" """
# 	for j in range(m):
# 		for k in range(j):
# 			# Linear combination of column j and k
# 			for i in range(k+1,m):
# 				A[i,j] = A[i,j] - A[i,k] * A[k,j]
# 				# record shapshot of A

# 		# Update below the diagonal
# 		for i in range(j+1,m):
# 			A[i,j] = A[i,j] / A[j,j]
# 			# record snapshot of A, ims


# def darve_LU_demo(A):
# 	""" """
# 	ims = []

# 	m = n = 64
# 	A = np.zeros( (m,n) )
# 	for i in range(m):
# 		for j in range(n):
# 			A[i,j] = np.cos( (np.pi*i*j) / n)


# 	fig = plt.figure()
# 	# Column-wise
# 	LU_column_up_looking(A,ims)
# 	LU_column_down_looking(A,ims)

# 	# Row-wise
# 	#LU_row_left_looking(A,ims)
# 	#LU_row_right_looking(A,ims)

# 	#plt.colorbar()
# 	ani = animation.ArtistAnimation(fig, ims, interval=1, repeat=False)
# 	plt.show()






def back_substitution(A,b):
	""" """
	n,_ = A.shape
	x = np.zeros((n,1))
	for k in range(n-1,-1,-1):
		for j in range(k+1,n):
			b[k] = b[k] - A[k,j] * x[j]
		x[k] = b[k] / A[k,k]

	return x


def back_substitution_demo():
	""" """
	n = 10
	A = np.random.randint(1,10,(n,n))
	A = np.triu(A)
	x = np.random.randint(1,10,(n,1))
	b = A.dot(x)

	x_est = back_substitution(A,b)
	# print('x_est ', x_est)
	# print('x ', x)
	print(x-x_est)



def LU(A):
	""" 
	Alternatively, we could compute multipliers,
	update the A matrix entries that change 
	(lower square block), and then store the
	multipliers in the empty, lower triangular part of A.
	"""
	n,_ = A.shape
	M = np.zeros((n-1,n,n))
	U = A.copy()
	# loop over the columns of U
	for k in range(n-1):
		M[k] = np.eye(n)
		# compute multipliers for each row under the diagonal
		for i in range(k+1,n):
			M[k,i,k] = -U[i,k] / U[k,k]
		# must update the matrix to compute
		# multipliers for next column
		pdb.set_trace()
		U = M[k].dot(U)

	L = np.eye(n)
	# left-multiply higher M matrices
	for k in range(n-1):
		L = M[k].dot(L)
	L = np.linalg.inv(L)
	return L,U


def LU_demo():
	""" """
	n = 10
	#A = np.random.randint(1,10,(n,n))

	A = np.array([[7, 5, 4, 6, 7, 1, 4, 1, 1, 2],
       [9, 1, 2, 2, 4, 8, 9, 5, 4, 5],
       [6, 5, 6, 2, 1, 5, 6, 2, 7, 4],
       [6, 8, 3, 6, 2, 5, 8, 4, 7, 3],
       [6, 7, 6, 7, 8, 4, 8, 7, 8, 8],
       [4, 4, 4, 4, 5, 2, 1, 7, 4, 2],
       [3, 7, 4, 9, 7, 5, 3, 8, 2, 3],
       [7, 1, 8, 8, 7, 6, 4, 8, 5, 8],
       [4, 5, 3, 5, 1, 4, 6, 4, 3, 3],
       [3, 3, 3, 3, 7, 4, 5, 2, 5, 9]])

	L,U = LU(A.copy())
	pdb.set_trace()


def Cholesky(A):
	""" 
	Only operate on bottom triangle of the matrix A,
	since A is symmetric (get same constraints
	from upper and lower triangle of A).
	"""
	n,_ = A.shape
	G = np.zeros((n,n))

	# populate lower triangular part
	for i in range(n):
		for j in range(n):
			if j > i:
				continue
			k_sum = 0
			if j==i: # diagonal element
				for k in range(j):
					k_sum += (G[j,k]**2)
				G[j,j] = np.sqrt(A[j,j] - k_sum)
			else:
				for k in range(j):
					k_sum += G[i,k]*G[j,k]
				G[i,j] = (A[i,j] - k_sum)/G[j,j]

	return G


def Cholesky_demo():

	for i in range(100):
		print('on i', i)
		n = 100
		# generate a random n x n matrix
		A = np.random.randn(n,n)

		# construct a symmetric matrix using either
		A = 0.5*(A+A.T)
		#A = A.dot(A.T)

		# make symmetric diagonally dominant matrix
		A += n * np.eye(n)
		w,v = np.linalg.eig(A)
		G = Cholesky(A)

		assert(np.allclose(np.linalg.cholesky(A),G))
		assert( (G.dot(G.T) - A ).sum() < 1e-10 )




if __name__ == '__main__':
	#main()
	#LU_demo()
	Cholesky_demo()



