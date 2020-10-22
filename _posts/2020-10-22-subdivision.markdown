

Eigenvalues of Subdivision Matrix
Rossignac + Scott Schaefer 2008 paper with , eigenvalues of the subdivision matrix are discussed.

Derivation (what is the subdivision matrix, and why are it's eigenvalues relevant), and why Laurent polynomials are useful?


Paper excerpt:

To establish the continuity of *Js for different values of s, we first consider the necessary conditions for continuity due to Reif [Rei95]. Given the subdivision matrix for Js, if the subdivision scheme produces curves that are Cm, then the eigenvalues of its subdivision matrix are of the form 1, (1/2), (1/4), ..., (1/2)m, λ, ... where λ<(1/2)m. The eigenvalues of the subdivision matrix for the Js subdivision are 1, (1/2), (1/4), (1/8), (2–s)/8, (s–1)/16, (s–1)/16, 0, 0. It is easy to verify that Js subdivision satisfies the necessary conditions for C1 continuity when –2<s<6, C2 continuity when 0<s<4, C3 continuity when 1<s<3, and C4 continuity when s=3/2. Notice that these conditions are only necessary, they are not sufficient.

To determine sufficient conditions on the subdivision scheme, we use the Laurent polynomial of the subdivision scheme given by

S(z) = (s–1)/16 + s/8 z + (9–s)/16 z2 + (1–s/4) z3 + (9–s)/16 z4 + s/8 z5 + (s–1)/16 z6,


## Derivation

6 consecutive points (A,B,C,D,E,F) of the a given level of subdivision of the original polygon may be used to compute 6 corresponding points (A',B',C',D',E',F') on the next subdivision level.

Each one of these (A',B',C',D',E',F') points new points is a weighted (affine) average of the (A,B,C,D,E,F) points. Hence, we can use a matrix M to represent this relation.

(A',B',C',D',E',F') = M * (A,B,C,D,E,F)

At the next subdivision level, you have 

(A",B",C",D",E",F") = M * (A',B',C',D',E',F') = M2 * (A,B,C,D,E,F)

and so on.

We want to compute where these 6 points converge.

So, we want M∞ 



## Implementation


```python
class pts:
	"""
	Class for manipulating and displaying 
	pointclouds or polyloops in 3D
	"""
	def __init__(self, G: np.ndarray = None) -> None:
		""" """
		self.maxnv: int = 16000 # max number of vertices
		self.G: np.ndarray = np.zeros((self.maxnv,3)) # geometry table (vertices)
		
		if G is None:
			self.nv: int = 0
		else:
			self.nv: int = G.shape[0]
			self.G[:self.nv] = G

		self.L: List[str] = [''] * self.maxnv # labels of points
		self.LL: np.ndarray = np.zeros((self.maxnv,3)) # displacement vectors
		self.pv: int = 0
		self.iv: int = 0
		self.dv: int = 0
		self.pp: int = 0

	def n(self, v: int) -> int:
		""" get next vertex """
		return (v+1) % self.nv
	
	def p(self, v: int) -> int:
		""" get prev vertex"""
		if(v==0):
			return self.nv-1
		else:
			return v-1

	def addPt(self, P: np.ndarray):
		""" appends a new point at the end
		Args:
			PNT

		Returns:
			pts
		"""
		self.G[self.nv] = P
		self.pv = self.nv
		self.L[self.nv] = 'f'
		self.nv += 1
		return self 


	def empty(self):
		self.nv = 0
		self.pv = 0

	def copyFrom(self, Q):
		"""
		Args:
			Q: pts

		Returns:
			pts
		"""
		self.empty()
		self.nv = Q.nv
		self.G = Q.G

		return self # set SELF as a clone of Q

	def subdivideQuinticInto(self, Q):
		"""
		Args:
			Q: pts
		"""
		s: float = 1.5
		Q.empty()
		for i in range(self.nv):
			Q.addPt(
				B(
					self.G[self.p(i)],
					self.G[i],
					self.G[self.n(i)],
					s
				)
			)
			Q.addPt(
				F(
					self.G[self.p(i)],
					self.G[i],
					self.G[self.n(i)],
					self.G[self.n(self.n(i))],
					s
				)
			)

		return self


def draw_JSpline():

	# pts P; // polyloop reference used for editing
	# pts O; // the other polyloop (not P)
	# pts Q = new pts(); // second polyloop in 3D
	# pts R = new pts(); // inbetweening polyloop L(P,t,Q);

	level = 10

	P1 = load_pts1()
	P2 = load_pts2()

	P = pts(P1)
	O = pts(P2)

	Q = pts()
	R = pts()

	plot_polyloop(P.G, color='g')
	plot_polyloop(O.G, color='r')

	# pdb.set_trace()

	pdb.set_trace()
	# Subdivide and display P
	R = R.copyFrom(P)
	for i in range(level):
		Q = Q.copyFrom(R)
		Q = Q.subdivideQuinticInto(R) # provide code

	plot_polyloop(Q.G, color='m')
	plt.show()

	pdb.set_trace()
	R.showMyProject() # Provided code (In TAB pts) for project 3
 
	# Subdivide and display O
	R.copyFrom(O)
	for i in range(level):
		Q.copyFrom(R)
		Q.subdivideQuinticInto(R) # provide code

	#R.showTube()
  ```
  
  ## References
  1. https://people.engr.tamu.edu/schaefer/research/js.pdf
  2. http://121.52.159.154:8080/jspui/bitstream/123456789/200/1/Complete%20Analysis%20of%203-point%20Binary.pdf 
  3. https://www.sciencedirect.com/science/article/pii/S0898122110000568

  
  
