
import numpy as np
import pdb
import matplotlib.pyplot as plt


def plot_2d_vectors(v1,v2,c1,c2):
	plt.arrow(0, 0, v1[0], v1[1], color=c1, head_width=.1, length_includes_head=True)
	plt.arrow(0, 0, v2[0], v2[1], color=c2, head_width=.1,length_includes_head=True)

def plot_circle_points():
	# generate n points along a circle
	n = 100
	radius = 1
	circ_x = np.array([np.cos(2*np.pi/n*x)*radius for x in range(0,n+1)])
	circ_y = np.array([np.sin(2*np.pi/n*x)*radius for x in range(0,n+1)])
	plt.plot(circ_x,circ_y, 'y--')


def classical_gs(A):
	""" 
	Classical Gram Schmidt (MGS). R will always be square.
	"A" must be full rank.

		Args:
		-	A:

		Returns:
		-	Q: Orthogonal matrix of shape (m,n)
		-	R: Upper triangular matrix of shape (n,n)
	"""
	m,n = A.shape
	Q = np.zeros((m,m))
	R = np.zeros((n,n))
	for k in range(n):
		for i in range(k):
			R[i,k] = Q[:,i].T.dot(A[:,k])
			A[:,k] -= R[i,k] * Q[:,i]
		R[k,k] = np.linalg.norm(A[:,k])
		Q[:,k] = A[:,k] / R[k,k]

	return Q,R

def modified_gs(A):
	""" 
	Modified Gram Schmidt (MGS). R will always be square.
	"A" must be full rank.

	Each time you orthogonalize you introduce some small error 
	in random directions. With classical GS, these errors all 
	add up; with modified GS they get eliminated.

	Compare: if you compute the subtraction of q0 and q1 
	separately from the k-th vector, they all introduce random error.

	On the other hand, if you first subtract q0 (times the 
	right coefficient), it introduces error in direction of q1, 
	but that error is then eliminated when you compute the 
	coefficient for subtracting q1.

		Args:
		-	A: Numpy array of shape (m,n)

		Returns:
		-	Q: Orthogonal matrix of shape (m,n)
		-	R: Upper triangular matrix of shape (n,n)
	"""
	m,n = A.shape
	R = np.zeros((n,n))
	for k in range(n):
		R[k,k] = np.linalg.norm(A[:,k])
		A[i,k] /= R[k,k]
		for j in range(k+1,n,1):
			R[k,j] = A[:,k].dot(A[:,j])

			# note that the columns a_k are updated online
			# before re-use, so errors are eliminated?
			A[:,j] = A[:,j] - A[:,k] * R[k,j]
	Q = A
	return Q,R


def unit_circle_gs():

	a1 = np.array([0.9,2.2])
	a2 = np.array([2.,1.9])
	plot_2d_vectors(a1,a2,'r','b')

	q1 = a1 / np.linalg.norm(a1)

	q2_tilde = a2 - (a2.T.dot(q1)) * (q1)
	q2 = q2_tilde / np.linalg.norm(q2_tilde)

	plot_2d_vectors(q1,q2,'g','m')
	plot_circle_points()
	plt.xlim([-3,3])
	plt.ylim([-3,3])
	plt.show()



if __name__ == '__main__':

	# unit_circle_gs()
	# quit()
	m = 10
	n = 10
	A = (np.random.randn(m,n)-1)*2

	plot_2d_vectors(A[:,0], A[:,1],'r','b')
	#Q,R = classical_gs(A)
	Q,R = modified_gs(A)
	plot_2d_vectors(Q[:,0],Q[:,1],'g','m')
	plot_circle_points()

	plt.xlim([-6,6])
	plt.ylim([-6,6])
	plt.show()

	pdb.set_trace()
	
	# pdb.set_trace()




