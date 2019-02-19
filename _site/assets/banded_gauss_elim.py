
import pdb
import numpy as np


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
	"""
	n,_ = A.shape

	w = np.zeros(n)
	x = np.ones(n)
	y = np.zeros(n)
	z = np.zeros(n)

	pdb.set_trace()

	y[0] = A[0,0]
	for i in range(0,n):

		if i > 0:
			y[i] = A[i,i] - w[i-1]*z[i-1]

		if i != n-1:
			z[i] = A[i,i+1]
			w[i] = A[i+1,i] / y[i]

		
	L = np.zeros((n,n))
	U = np.zeros((n,n))

	for i in range(n):
		for j in range(n):
			# diagonal
			if i==j:
				L[i,j] = x[i]
				U[i,j] = y[i]

			# superdiagonal
			if i==j-1:
				U[i,j] = z[i]

			# subdiagonal
			if i==j+1:
				L[i,j] = w[i]

	return L,U


def main():
	n=4
	q = 1
	p = 1

	A = np.random.randint(1,10,(n,n))

	A = np.array([	[8,7,0,0],
					[7,3,2,0],
					[0,1,2,6],
					[0,0,9,5]])

	for i in range(n):
		for j in range(n):
			if (j > i + 1) or (i > j + 1):
				A[i,j] = 0

	L,U = LU(A)


	# for k in range(1,n):
	# 	for i in range(k+1,min(k+p,n)):
	# 		A[i,k] = A[i,k] / A[k,k]
	# 	for j in range(k+1,min(k+1,n)):
	# 		for i in range(k+1,min(k+p,n)):
	# 			A[i,j] = A[i,j] - A[i,k] * A[k,j]

	# L = np.tril(A)
	# U = np.triu(A)

	# for i in range(n):
	# 	L[i,i] = 1

	print(np.round(L.dot(U)))
	print(A)




if __name__ == '__main__':

	main()


