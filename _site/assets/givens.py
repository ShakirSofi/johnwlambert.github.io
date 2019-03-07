
import numpy as np
import pdb



def givens(A,i,j):
	""" """
	
	t = x2/x1
	c = 1./np.sqrt(1+np.square(t))
	s = c * t

	G[i,j] = c
	aaa = s
	bbb = -s
	ccc = c


def givens_demo():
	""" """
	n = 4
	A = np.random.randint(-10,10,(n,n))


	givens(A,i,j)


def fast_givens():
	""" """
	x1 = 101.
	x2 = 73.

	x = np.array([x1,x2]).reshape(2,1)

	d1 = 9.
	d2 = 53.

	D = np.array([[d1, 0],[0,d2]])
	
	# beta1 = (x1/x2)*(d2/d1)
	# alpha1 = -(x1/x2)

	beta1 = -(x2/x1)
	alpha1 = (x2/x1)*(d2/d1)

	# M1 = np.array([	[beta1,1.],
	# 				[1.,alpha1]])
	M1 = np.array([	[1., alpha1,],
					[beta1, 1.]])

	Dtilde = M1.T.dot(D).dot(M1)
	y = M1.dot(x)
	print(y)
	print(np.round(Dtilde,2))


def main():
	fast_givens()


if __name__ == '__main__':
	main()