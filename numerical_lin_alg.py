
import numpy as np
import matplotlib.pyplot as plt


def QR_householder_demo():

	# Initialize matrix
	m = n = 32
	Z = np.zeros( (m,n) )
	for i in range(m):
		for j in range(n):
			A[i,j] = np.cos( (np.pi*i*j) / n)

	# Apply sequence of Householder transformations
	for k in range(n-1):
		v,beta = house(A[k:,k])
		A[k:,k:] = A[k:,k:] - beta * outer(v, v.dot(A[k:,k:]))
		# record a snapshot of A


def gram_schmidt_demo():
	""" """
	m = n = 64
	A = np.zeros( (m,n) )
	for i in range(m):
		for j in range(n):
			A[i,j] = np.cos( (np.pi*i*j) / n)

	B = A.copy()

	# Apply the Modified Gram-Schmidt orthogonalization process
	for k in range(n):
		# Make column k orthogonal to all previous columns
		for r in range(k):
			A[:,k] = A[:,k] - A[:,r].dot(A[:,k]) * A[:,r]
		# Normalize column
		A[:,k] = A[:,k] / np.linalg.norm( A[:,k] )





def LU_column_up_looking(A,ims):
	""" """
	for j in range(m):
		# Update above the diagonal
		for i in range(1,j+1):
			# Dot product
			for k in range(i):
				A[i,j] = A[i,j] - A[i,k] * A[k,j]
			# record snapshot of A

		# Update below the diagonal
		for i in range(j+1,m):
			# Dot product
			for k in range(j):
				A[i,j] = A[i,j] - A[i,k] * A[k,j]




def LU_column_down_looking(A,ims):
	""" """
	for j in range(m):
		for k in range(j):
			# Linear combination of column j and k
			for i in range(k+1,m):
				A[i,j] = A[i,j] - A[i,k] * A[k,j]
				# record shapshot of A

		# Update below the diagonal
		for i in range(j+1,m):
			A[i,j] = A[i,j] / A[j,j]
			# record snapshot of A, ims


def LU_demo():
	""" """
	ims = []

	# Column-wise
	LU_column_up_looking(A,ims)
	LU_column_down_looking(A,ims)

	# Row-wise
	LU_row_left_looking(A,ims)
	#LU_row_right_looking(A,ims)

	plt.colorbar()
	ani = an.ArtistAnimation(fig, ims, interval=1, repeat=False)
	plt.show()


def Cholesky(A,ims):
	""" """
	for j in range(m):
		for i in range(j,m):
			for k in range(j):
				A[i,j] = A[i,j] - A[i,k] * A[j,k]
		# The usual equation is
		# A[i,j] = A[i,j] - A[i,k] * A[k,j]
		# But we use the symmetry of A to only access the lower
		# triangular part of A.
		record_snapshot(A, ims)
		piv = math.sqrt(A[j,j])
		for i in range(j,m):
			A[i,j] = A[i,j] / piv




def main():
	""" """
	LU_demo()




if __name__ == '__main__':
	main()


