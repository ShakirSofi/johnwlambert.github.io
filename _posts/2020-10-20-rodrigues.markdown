---
layout: post
comments: true
permalink: /rodrigues/
title:  "Rodrigues' Formula"
excerpt: "Including Trust-Region Variant (Levenberg-Marquardt)"
date:   2020-10-19 11:00:00
mathjax: true
---


## Derivation

### Implementation

```python
def Rodrigues(v: np.ndarray, k: np.ndarray, theta: float):
	""" rotate vector v about axis k (unit vector) """
	k /= np.linalg.norm(k)

	c = np.cos(theta)
	s = np.sin(theta)
	v_rot = v*c + np.cross(k,v)*s + k*(k.dot(v))*(1-c)
	return v_rot
```

<div class="fig figcenter fighighlight">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Rodrigues-formula.svg/600px-Rodrigues-formula.svg.png" width="65%">
  <img src="https://user-images.githubusercontent.com/16724970/96642754-966d5580-12f4-11eb-8ea0-45b8a023ab42.jpg" width="65%">
</div>

```python
	# z points up
	v = np.array([-2.,0,1.])
	k = np.array([0,0,1.])
	theta = np.deg2rad(90 + 45)
	pdb.set_trace()
	v_rot = Rodrigues(v,k,theta)
```

Install Mayavi per instructions [here](https://docs.enthought.com/mayavi/mayavi/installation.html).

## More Derivations

\end{itemize}
\subsection{June 5: Rodrigues}
\begin{itemize}
\item 
As we have seen before, we can simulate with a first order approximation

\begin{equation}
\begin{bmatrix}
\phi_{t + \delta t} \\
\theta_{t + \delta t} \\
\psi_{t + \delta t} \\
\end{bmatrix} \approx \begin{bmatrix} \phi_t \\ \theta_t \\ \psi_t \end{bmatrix} + \delta_t f(\phi, \theta, \psi, \vec{\omega}) + w_t
\end{equation}
where $w_t$ is your process noise (whitened Gaussian) and we can use EKF, UKF, PF

\end{itemize}
\subsection{Euler's Rotation Theorem and Angle-Axis Representation}
\begin{itemize}

\item Rotation about a specific axis $\vec{a}$, through some angle $\phi$

\begin{equation}
axis = \vec{a} = \begin{bmatrix} a_x \\ a_y  \\ a_z \end{bmatrix}
\end{equation}

\item Static notion of orientation (not obvious how you got to that new orientation)
\item But there is always an angle and an axis to get there


\item How do we relate $\vec{a}$ and $\phi$?
\item We have $\|\vec{a}\| = 1$ because it is just a direction
\item What is the function
$$
\leftidx{^w}R_b = f(\vec{a}, \phi)
$$
\item Rodrigues' Formula: (you give me angle and axis, I give you the rotation matrix it represents)

$$
\leftidx{^w}R_b = I + \vec{a}^{\wedge} \mbox{sin}(\phi) + \Big(1 - \mbox{cos}(\phi) \Big) (\vec{a}^{\wedge})^2
$$

where
\begin{equation}
a^{\wedge} = \vec{a} \times (\cdots) =  \begin{bmatrix}
0 & -a_z & a_y \\ a_z & 0 & -a_x \\ -a_y &  a_x & 0
\end{bmatrix}
\end{equation}
\item Motion from initial frame to final frame.


\item Consider a point fixed in the body $\vec{p}_t$, a function of time. 
\item Initially, in body frame
\begin{equation}
\vec{p}(0) = \vec{p}_b
\end{equation}
we will end up at
\begin{equation}
\vec{p}(T) = \vec{p}_w
\end{equation}
(in the world frame)
\item Kinematics of this point:

Start in the body frame $F_b$
\begin{equation}
\vec{p} = (\frac{ \partial \vec{p} }{ \partial t})_{F_b} + \vec{w} \times \vec{p}
\end{equation}
\item Since point is fixed in the body,

\begin{equation}
(\frac{ \partial \vec{p} }{ \partial t})_{F_b} = 0
\end{equation}
where 
\begin{equation}
\vec{p} = \Omega \vec{p} = \omega^{\wedge} \vec{p}
\end{equation}

\item $\omega$ is your rotation rate (angular velocity). We let 
\begin{equation}
\frac{\vec{w}}{\| \vec{\omega} \|} = \vec{a}
\end{equation}
This is the direction of the angular velocity. We want 
\begin{equation}
\| \vec{\omega} \| T = \phi
\end{equation}
so 
\begin{equation}
\vec{w} = \frac{ \vec{a}\phi }{T}
\end{equation}

\item The Diff. Eqn (ODE) in $\vec{p}$:

\begin{equation}
\dot{\vec{p}} = \vec{a}^{\wedge} \frac{\phi}{T} \vec{p}
\end{equation}
\item Linear ODE
\item 

\begin{equation}
\dot{\vec{p}} = A \vec{p}
\end{equation}

Solution via the matrix exponential
\begin{equation}
\vec{p}(t) = e^{At} \vec{p}(0)
\end{equation}





\begin{equation}
\vec{p}(T) = \vec{p}_w = \mbox{exp}( \vec{a}^{\wedge} \frac{\phi}{T}T ) \vec{p}_b
\end{equation}

\item Related through the exponential map, simplifying to 

\begin{equation}
\vec{p}(T) = \vec{p}_w = \mbox{exp}( \vec{a}^{\wedge} \phi ) \vec{p}_b
\end{equation}
\item 
$$
\leftidx{^w}R_b = \mbox{exp}(\vec{a}^{\wedge} \phi)
$$
where $(\vec{a}, \phi)$ are in exponential coordinates
\item Taylor Series expansion of this guy, evaluated at 0, along with some work, will get us 

\begin{equation}
e^{\vec{a}^{\wedge} \phi} = I + (\vec{a}^{\wedge} \phi) + \frac{1}{2!}(\vec{a}^{\wedge} \phi)^2 + \frac{1}{3!}(\vec{a}^{\wedge} \phi)^3 + \dots
\end{equation}

\item Try to put exponential series into sine and cosine (odd -sin and even -cosine terms)

\item Note:

\begin{equation}
(\vec{a}^{\wedge})^2 = \vec{a} \vec{a}^T - I
\end{equation}

\textbf{Proof by Demonstration}:

\item When we expand this out, we find
\begin{equation}
(\vec{a}^{\wedge})^2 = \begin{bmatrix} 0 & -a_z & a_y \\ a_z & 0  &-a_x \\ -a_y & a_x & 0 \end{bmatrix}
\begin{bmatrix} 0 & -a_z & a_y \\ a_z & 0  &-a_x \\ -a_y & a_x & 0 \end{bmatrix}
\end{equation}

and when we expand out $\vec{a}\vec{a}^T -I$ we see it's identical since $\vec{a}$ is a unit vector.

$\vec{a}\vec{a}^T \vec{a}^{\wedge}$
\item 
Take the transpose of the whole thing,
$\Bigg((\vec{a}^{\wedge})^T \vec{a}\vec{a}^T\Bigg)^T$

and a skew symmetric matrix is a cross product.
$\vec{a}^{\wedge} = (\vec{a} \times \vec{a}$

\begin{equation}
(\vec{a}^{\wedge})^3 = - \vec{a}^{\wedge}
\end{equation}

Thus we end up that every even power of $\vec{a}^{\wedge})^i = \pm (\vec{a})^2$
\item Then any odd power

\begin{equation}
(\vec{a}^{\wedge})^i \pm \vec{a}^{\wedge}
\end{equation}


\item odds:
\begin{equation}
\phi - \frac{1}{3!}\phi^3 + \frac{1}{5!} \phi^5 - \frac{1}{7}\phi^7) \vec{a}^{\wedge}
\end{equation}

\item evens
\begin{equation}
\frac{1}{2!}\phi^2 - \frac{1}{4!} \phi^4 + \frac{1}{6}\phi^6 -\frac{1}{8}\phi^8 ) (\vec{a}^{\wedge})^2
\end{equation}
This reduces to $(1-\mbox{cos} \phi)$


\end{itemize}
\subsection{How to use Angle-Axis }

\begin{equation}
\phi = \mbox{cos}^{-1} \Bigg(  \frac{ \mbox{Tr}() }{2}\Bigg)
\end{equation}

\begin{equation}
\vec{a} = \frac{1}{2 \mbox{sin} \phi} \begin{bmatrix}
(R_b^w)_{32} - (R_b^w)_{23} \\
(R_b^w)_{13} - (R_b^w)_{31} \\ 
(R_b^w)_{21} - (R_b^w)_{12} \\
\end{bmatrix}
\end{equation}



```python

import pdb
from typing import List, NamedTuple

import matplotlib.pyplot as plt
import numpy as np

import mayavi
from mayavi import mlab
from argoverse.visualization.mayavi_utils import (
	draw_mayavi_line_segment,
	draw_coordinate_frame_at_origin
)

"""
SUBDIVISION: THESE ARE INCORRECT: students can fix and use
"""
def B(A, B, C, s: float):
	"""
	Args:
		A: PNT
		B: PNT
		C: PNT

	Returns:
		B: a tucked B towards its neighbors (PNT)"""
	P2j = (s*A + (8-2*s)*B + s*C)
	return P2j / 8

def F(
	A: np.ndarray,
	B: np.ndarray,
	C: np.ndarray,
	D: np.ndarray,
	s: float
) -> np.ndarray:
	"""
	Args:
		A: PNT
		B: PNT
		C: PNT
		D: PNT

	Returns:
		PNT: returns a bulged mid-edge point 
	"""
	P2j_ = (s-1)*A + (9-s)*B + (9-s)*C + (s-1)*D
	return P2j_ / 16

	# using PNT P(PNT A, PNT B) {return P((A.x+B.x)/2.0,(A.y+B.y)/2.0,(A.z+B.z)/2.0); }                             // (A+B)/2


####################################

def load_pts1():
	""" """
	return np.array(
		[
			[486.82037,-36.00861,192.40654],
			[808.35504,-395.299,113.95366],
			[587.1747,-609.05273,334.06415],
			[113.0201,-721.1454,312.5196],
			[-226.97775,-585.527,396.82755],
			[-112.4024,197.44191,134.78064],
			[-402.18472,526.12823,449.79633],
			[-242.99121,740.1481,299.81494],
			[284.4596,852.22266,335.347],
			[567.863,671.8136,391.55554],
			[596.9898,411.42258,484.11658],
			[392.7334,183.86339,249.68994],
			[-3.4672976,-206.9228,255.67894],
			[414.87363,-276.92725,267.26868],
			[302.3617,-120.64748,199.78485]
		])


def load_pts2():
	""" """
	return np.array(
		[
			[710.51544,-126.12543,1.2397095],
			[789.6114,-202.04018,353.12115],
			[587.1747,-609.05273,198.82367],
			[10.696083,-784.35486,177.27911],
			[-290.6693,-658.7666,261.58707],
			[-295.6774,-149.64072,449.395],
			[-109.80004,326.68582,314.55585],
			[-242.99121,740.1481,164.57446],
			[90.46646,1025.7959,357.98865],
			[614.851,918.7967,758.326],
			[392.7334,183.86339,114.449455],
			[220.40666,56.137333,158.83093],
			[396.11987,-55.51138,401.9234],
			[277.95963,-253.40622,285.94055],
			[594.0142,-305.75613,64.544365]
		])



def plot_polyloop(polyline: np.ndarray, color: str):
	""" Accept Nx3 point array, form polyloop """
	pl = np.vstack([polyline,polyline[0]])
	plt.scatter(pl[:,0], pl[:,1], 10, c=color, marker='.')
	plt.plot(pl[:,0], pl[:,1], c=color)


def cross(U, V):
	""" UxV cross product (normal to both)
	Args:
		U: VCT
		V: VCT
	"""
	Ux, Uy, Uz = U
	Vx, Vy, Vz = V

	return np.array([
		Uy*Vz-Uz*Vy,
		Uz*Vx-Ux*Vz,
		Ux*Vy-Uy*Vx
	])                  


def twistFreeQuadMesh(points):
	"""
	 twist-free normal propagation
	  constant-radius tubes, each as a twist-free quad-mesh with checkerboard coloring.
	   4 quads, 5 quads with stripes instead of a checkerboard pattern, and 6quads
	
	Args:
		points: polyloop vertices
	"""

	#divide unit circle into 6 points
	#form quad from corresponding points

	fig = mayavi.mlab.figure(figure=None, bgcolor=(1, 1, 1), fgcolor=None, engine=None, size=(1600, 1000))

	draw_mayavi_line_segment(
	    fig,
	    points
	)

	n_pts = points.shape[0]
	Y = np.zeros((n_pts,))
	T = np.zeros((n_pts,))
	N = np.zeros((n_pts,))
	for i in range(n_pts):

		A = points[i]
		B = points[(i+1) % n_pts]
		C = points[(i+2) % n_pts]

		AB = normalize(B - A)
		BA = normalize(A - B)
		BC = normalize(C - B)
		N[i] = cross(BA,BC)
		N[i] = normalize(N[i])
		draw_mayavi_line_segment(
		    fig,
		    [points[i], points[i] + N[i]],
		    color=(0,1,0),
		    line_width = 10
		)

		T[i] = cross(AB,N)
		draw_mayavi_line_segment(
		    fig,
		    [points[i], points[i] + T[i]],
		    color=(0,0,1),
		    line_width = 10
		)

		x = Y[i-1].dot(T[i-1])
		y = Y[i-1].dot(N[i])
		Y[i] = x * T[i] + y * N[i]
		draw_mayavi_line_segment(
		    fig,
		    [points[i], points[i] + Y[i]],
		    color=(0,0,1),
		    line_width = 10
		)

	fig = draw_coordinate_frame_at_origin(fig)
	#l = mayavi.mlab.plot3d(x, y, z, np.sin(mu), tube_radius=0.025, colormap="Spectral")
	mayavi.mlab.show()


class VectorFrame:
	def __init__(self, o,t,n):
		self.o = o # origin
		self.t = t # tangent
		self.r = n # rotational axis
		self.n = normalize(cross(t,n))



def getOrthonormalFrame(points: np.ndarray, i: int):
	""" """
	n_pts = points.shape[0]
	A = points[i]
	B = points[(i+1) % n_pts]
	C = points[(i+2) % n_pts]

	AB = normalize(B - A)
	BA = normalize(A - B)
	BC = normalize(C - B)
	N = cross(BA,BC)
	N = normalize(N)
	return VectorFrame(o=A,t=AB,n=N)


def getRotationMinimizingFrame(points: np.ndarray):
	""" """
	n_pts = points.shape[0]

	# Start off with the standard tangent/axis/normal frame
	# associated with the curve
	frames = [getOrthonormalFrame(points, -1)]

	# start constructing RM frames
	for i in range(n_pts):
		# start with the previous, known frame
		x0 = frames[-1]

		# get the next frame
		# we're going to throw away its axis and normal
		x1 = getOrthonormalFrame(points, i)

		# First we reflect x0's tangent and axis onto x1, through
		# the plane of reflection at the point midway x0--x1
		v1 = x1.o - x0.o
		c1 = v1.dot(v1)
		riL = x0.r - v1 * (2/c1) * v1.dot(x0.r)
		tiL = x0.t - v1 * (2/c1) * v1.dot(x0.t)

		# Then we reflect a second time, over a plane at x1
		# so that the frame tangent is aligned with the curve tangent:
		v2 = x1.t - tiL
		c2 = v2.dot(v2)

		x1.r = riL - v2 * (2/c2) * v2.dot(riL)
		x1.n = cross(x1.r,x1.t)

		# we record that frame, and move on
		frames += [x1]

	# and before we return, we throw away the very first frame,
	frames = frames[1:]

	angle_diff = get_angle_between(frames[0].r, frames[-1].r)
	if not np.allclose(Rodrigues(frames[-1].r, frames[-1].t, angle_diff), frames[0].r):
		angle_diff = np.pi*2 - angle_diff
	print('Angle diff: ', np.rad2deg(angle_diff))
	num_frames = len(frames)
	theta = angle_diff / (num_frames - 1)
	# remove twists between end-point and start-point
	for i, frame in enumerate(frames):
		frames[i].r = Rodrigues(v=frame.r,k=frame.t,theta=theta*i)
		frames[i].n = Rodrigues(v=frame.n,k=frame.t,theta=theta*i)

	fig = mayavi.mlab.figure(figure=None, bgcolor=(1, 1, 1), fgcolor=None, engine=None, size=(1600, 1000))
	draw_mayavi_line_segment(
	    fig,
	    points
	)

	draw_coordinate_frames = True
	if draw_coordinate_frames:
		for frame in frames:
			draw_mayavi_line_segment(
			    fig,
			    [frame.o, frame.o + frame.t],
			    color=(1,0,0),
			    line_width = 10
			)
			draw_mayavi_line_segment(
			    fig,
			    [frame.o, frame.o + frame.r],
			    color=(0,1,0),
			    line_width = 10
			)		
			draw_mayavi_line_segment(
			    fig,
			    [frame.o, frame.o + frame.n],
			    color=(0,0,1),
			    line_width = 10
			)
	else:
		# draw the hexagon quad mesh
		for frame in frames:
			points = get_ngon_points(frame.o, frame.n, frame.r, n=6, radius=1)
			draw_mayavi_line_segment(
			    fig,
			    np.vstack([points,points[0]])
			)

	fig = draw_coordinate_frame_at_origin(fig)
	mayavi.mlab.show()

	return frames


def get_angle_between(a,b):
	""" Get angle between two 3d vectors, account for CW vs. CCW orientation"""
	angle_diff = np.arccos(a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b)) )
	return angle_diff 


def test_get_angle_between():

	a = np.array([1,0,0])
	b = np.array([0,1,0])
	theta_rad = get_angle_between(a,b)
	theta_deg = np.rad2deg(theta_rad)
	assert np.isclose(theta_deg, 90)

	a = np.array([1,0,0])
	b = np.array([-1,-1,0])
	theta_rad = get_angle_between(a,b)
	theta_deg = np.rad2deg(theta_rad)
	assert np.isclose(theta_deg, 225)
	pdb.set_trace()


def normalize(v: np.ndarray) -> np.ndarray:
	assert v.size == 3
	return v / np.linalg.norm(v)


def test_twistFreeQuadMesh1():
	""" """
	# vertices of polyloop
	polyloop = np.array(
		[
			[3,0,0],
			[5,0,0],
			[7,-2,0],
			[3,0,0]
		])
	twistFreeQuadMesh(polyloop)

def test_twistFreeQuadMesh2():
	""" """
	polyloop = load_pts1_JSpline4() / 100
	twistFreeQuadMesh(polyloop)



def test_twistFreeQuadMesh3():
	""" """
	polyloop = load_pts1_JSpline4() / 100
	#polyloop = load_pts2_JSpline5() / 100
	#polyloop = load_pts2_JSpline3() / 100
	getRotationMinimizingFrame(polyloop)



def load_pts1_JSpline3():
	return np.array(
		[
			[515.90125 , -134.41925 , 178.68292],
			[572.69543 , -175.80655 , 170.50702],
			[625.40436 , -231.97215 , 165.62088],
			[665.8761 , -297.14185 , 167.44434],
			[686.83704 , -364.89984 , 178.59827],
			[683.34644 , -429.7468 , 199.52708],
			[654.2506 , -488.65726 , 227.1211],
			[601.3314 , -540.27576 , 256.47183],
			[528.4545 , -584.11304 , 282.62775],
			[440.7178 , -619.7417 , 302.34943],
			[343.59955 , -645.99274 , 315.86517],
			[242.79825 , -660.8927 , 325.22302],
			[144.07219 , -661.6023 , 332.64294],
			[53.07941 , -644.35333 , 338.86893],
			[-24.782658 , -604.3875 , 341.52106],
			[-85.463776 , -537.66943 , 337.21997],
			[-127.347404 , -442.6007 , 323.7109],
			[-152.33727 , -321.73343 , 301.98853],
			[-166.94391 , -183.48413 , 278.4209],
			[-179.4609 , -39.26005 , 261.48688],
			[-197.14093 , 99.41425 , 258.5132],
			[-223.37198 , 223.7706 , 272.41183],
			[-254.85349 , 330.67065 , 298.41705],
			[-284.18073 , 420.87933 , 327.65784],
			[-302.42896 , 497.33786 , 350.73004],
			[-301.7378 , 563.43744 , 361.26895],
			[-277.89554 , 621.29224 , 359.52136],
			[-229.94601 , 671.8601 , 349.8966],
			[-159.79567 , 715.06366 , 338.5172],
			[-71.820526 , 749.9107 , 330.77008],
			[27.526917 , 774.615 , 328.85696],
			[130.82475 , 787.18854 , 332.76736],
			[230.78876 , 786.0333 , 341.25073],
			[321.37946 , 770.53394 , 352.78918],
			[398.9092 , 741.6489 , 366.57004],
			[461.48468 , 701.4206 , 381.93915],
			[508.44928 , 652.4837 , 397.85486],
			[539.8255 , 597.5744 , 412.34125],
			[555.7575 , 539.03955 , 421.94196],
			[556.5006 , 478.63232 , 422.99908],
			[542.411 , 417.30777 , 412.93243],
			[513.9356 , 355.01886 , 391.5183],
			[471.6015 , 290.512 , 362.16882],
			[416.93842 , 222.29279 , 330.23953],
			[353.40103 , 149.59183 , 301.33685],
			[287.29144 , 73.330475 , 279.62598],
			[228.68134 , -2.9134274 , 266.13846],
			[188.1731 , -74.13457 , 259.70215],
			[173.66034 , -135.12927 , 257.87134],
			[187.08885 , -181.72461 , 257.8568],
			[221.2174 , -212.00735 , 257.45575],
			[264.03674 , -225.95793 , 255.12393],
			[303.18866 , -225.08386 , 250.04745],
			[330.3852 , -212.05359 , 242.21494],
			[345.82724 , -190.3303 , 232.48944],
			[355.11816 , -164.04568 , 222.12576],
			[366.17676 , -137.87402 , 212.28795],
			[386.15033 , -116.90614 , 203.56647],
			[418.32794 , -106.523285 , 195.49582],
			[462.59723 , -111.38541 , 187.30743]
		])



def load_pts1_JSpline4():
	return np.array(
		[
			[516.55566 , -137.86052 , 178.76704],
			[544.2798 , -156.14827 , 174.71179],
			[571.92944 , -178.57748 , 171.12386],
			[598.5398 , -204.63254 , 168.37643],
			[623.10986 , -233.66042 , 166.87892],
			[644.6481 , -264.91928 , 167.03386],
			[662.2178 , -297.62714 , 169.1938],
			[674.9828 , -331.01074 , 173.61835],
			[682.2523 , -364.354 , 180.43106],
			[683.52747 , -397.04684 , 189.57643],
			[678.5454 , -428.63367 , 200.77682],
			[667.25385 , -458.78864 , 213.58728],
			[649.7837 , -487.29 , 227.45049],
			[626.42285 , -513.9955 , 241.75153],
			[597.58936 , -538.8168 , 255.8728],
			[563.80493 , -561.6947 , 269.24887],
			[525.66833 , -582.5739 , 281.42133],
			[483.8286 , -601.3778 , 292.09357],
			[438.95874 , -617.9834 , 301.1858],
			[391.75043 , -632.2194 , 308.78345],
			[342.909 , -643.8644 , 315.08557],
			[293.14865 , -652.64453 , 320.3536],
			[243.18735 , -658.232 , 324.85968],
			[193.74173 , -660.24286 , 328.8351],
			[145.52219 , -658.23505 , 332.41907],
			[99.2278 , -651.70667 , 335.60693],
			[55.54142 , -640.0939 , 338.19882],
			[15.095616 , -622.8245 , 339.86603],
			[-21.56123 , -599.37146 , 340.21732],
			[-53.998894 , -569.3065 , 338.86548],
			[-81.93925 , -532.3537 , 335.49347],
			[-105.29024 , -488.4429 , 329.92102],
			[-124.179825 , -437.76346 , 322.1709],
			[-138.98993 , -380.8177 , 312.5354],
			[-150.39041 , -318.4743 , 301.64255],
			[-159.25082 , -251.87888 , 290.35437],
			[-166.5521 , -182.36386 , 279.6647],
			[-173.29845 , -111.35881 , 270.59744],
			[-180.42899 , -40.300636 , 264.10443],
			[-188.72949 , 29.456234 , 260.96356],
			[-198.74426 , 96.72964 , 261.67676],
			[-210.68774 , 160.59949 , 266.3681],
			[-224.35645 , 220.49754 , 274.68182],
			[-239.20947 , 276.1535 , 285.8939],
			[-254.44957 , 327.54102 , 299.0237],
			[-269.10358 , 374.82367 , 312.9458],
			[-282.1034 , 418.3012 , 326.50122],
			[-292.36676 , 458.35516 , 338.6095],
			[-298.8778 , 495.3955 , 348.38007],
			[-300.76807 , 529.8063 , 355.22388],
			[-297.39697 , 561.89154 , 358.9652],
			[-288.33984 , 591.8795 , 359.765],
			[-273.37543 , 619.92596 , 358.04437],
			[-252.47365 , 646.1183 , 354.408],
			[-225.78336 , 670.4793 , 349.5676],
			[-193.62004 , 692.9706 , 344.2656],
			[-156.45352 , 713.4968 , 339.19827],
			[-114.89569 , 731.90906 , 334.93945],
			[-69.68822 , 748.0089 , 331.86395],
			[-21.667969 , 761.5668 , 330.17783],
			[28.267614 , 772.3405 , 329.94885],
			[79.19511 , 780.0936 , 331.13702],
			[130.19966 , 784.6144 , 333.6248],
			[180.40965 , 785.73364 , 337.2474],
			[229.03128 , 783.34375 , 341.82358],
			[275.38303 , 777.4171 , 347.1855],
			[318.9305 , 768.0242 , 353.20966],
			[359.26886 , 755.31866 , 359.79932],
			[396.10526 , 739.52203 , 366.86783],
			[429.24176 , 720.90814 , 374.32132],
			[458.55762 , 699.7877 , 382.04163],
			[483.992 , 676.49335 , 389.86942],
			[505.52646 , 651.3639 , 397.58685],
			[523.1676 , 624.7291 , 404.9007],
			[536.9297 , 596.89465 , 411.42517],
			[546.8342 , 568.13513 , 416.72192],
			[552.9096 , 538.6885 , 420.34003],
			[555.19086 , 508.74878 , 421.85593],
			[553.7195 , 478.46033 , 420.9134],
			[548.5427 , 447.91125 , 417.26355],
			[539.7137 , 417.12695 , 410.80475],
			[527.29065 , 386.06387 , 401.62277],
			[511.33707 , 354.603 , 390.03043],
			[491.9502 , 322.58008 , 376.51495],
			[469.2898 , 289.81592 , 361.68512],
			[443.60736 , 256.14633 , 346.21814],
			[415.27448 , 221.45245 , 330.807],
			[384.812 , 185.691 , 316.1075],
			[352.91876 , 148.92426 , 302.68533],
			[320.5002 , 111.35043 , 290.96313],
			[288.6976 , 73.33375 , 281.16788],
			[258.78644 , 35.36603 , 273.35956],
			[232.07544 , -1.9716597 , 267.46057],
			[209.80527 , -38.047462 , 263.28458],
			[193.04724 , -72.21711 , 260.56567],
			[182.60223 , -103.862366 , 258.98743],
			[178.89932 , -132.4294 , 258.21188],
			[181.89464 , -157.4672 , 257.90875],
			[190.97011 , -178.66599 , 257.78433],
			[205.0716 , -195.84584 , 257.58386],
			[222.84692 , -208.94507 , 257.09375],
			[242.78406 , -218.00899 , 256.14374],
			[263.3491 , -223.17831 , 254.6093],
			[283.1245 , -224.67773 , 252.41379],
			[300.94702 , -222.80458 , 249.5307],
			[316.046 , -217.91718 , 245.98593],
			[328.18124 , -210.42365 , 241.86],
			[337.54666 , -200.77774 , 237.27309],
			[344.67392 , -189.47507 , 232.3698],
			[350.33572 , -177.04898 , 227.30408],
			[355.4496 , -164.06686 , 222.22435],
			[360.9813 , -151.126 , 217.25818],
			[367.84833 , -138.84973 , 212.49725],
			[376.82352 , -127.883484 , 207.98244],
			[388.4386 , -118.89083 , 203.68849],
			[402.99835 , -112.5219 , 199.54779],
			[420.59515 , -109.38173 , 195.47375],
			[441.1228 , -109.99862 , 191.38431],
			[464.29123 , -114.7926 , 187.22565],
			[489.64062 , -124.04375 , 182.99557]
		])


def load_pts2_JSpline3():
	""" """
	return np.array(
		[
			[698.1858 , -185.84529 , 94.35156],
			[709.17737 , -183.71832 , 110.71509],
			[718.4133 , -185.02779 , 129.75238],
			[725.6819 , -190.16574 , 150.46677],
			[730.70605 , -199.40988 , 171.74101],
			[733.1609 , -212.91669 , 192.4209],
			[732.69006 , -230.71426 , 211.39885],
			[728.9235 , -252.6957 , 227.69743],
			[721.49396 , -278.61203 , 240.55304],
			[710.0541 , -308.0653 , 249.49939],
			[694.29376 , -340.50186 , 254.45114],
			[673.95764 , -375.24652 , 255.63591],
			[648.8636 , -411.53754 , 253.52641],
			[618.92053 , -448.56085 , 248.77252],
			[584.14624 , -485.48474 , 242.13322],
			[544.6858 , -521.4944 , 234.40878],
			[500.82907 , -555.8264 , 226.3728],
			[453.02917 , -587.80334 , 218.70424],
			[401.92023 , -616.86835 , 211.9195],
			[348.27567 , -642.56946 , 206.39352],
			[292.96637 , -664.5444 , 202.38078],
			[236.91893 , -682.50494 , 200.03644],
			[181.07379 , -696.2214 , 199.43738],
			[126.34346 , -705.5074 , 200.60324],
			[73.570816 , -710.2041 , 203.51752],
			[23.487175 , -710.1648 , 208.1487],
			[-23.329414 , -705.2396 , 214.47116],
			[-66.43893 , -695.29114 , 222.45169],
			[-105.524025 , -680.2096 , 232.03563],
			[-140.37482 , -659.9292 , 243.13327],
			[-170.87349 , -634.4432 , 255.6062],
			[-196.9791 , -603.8198 , 269.25366],
			[-218.71214 , -568.2178 , 283.79858],
			[-236.13943 , -527.9021 , 298.87433],
			[-249.3587 , -483.2596 , 314.01068],
			[-258.50708 , -434.77185 , 328.66934],
			[-263.76962 , -382.98816 , 342.2791],
			[-265.38763 , -328.49878 , 354.27133],
			[-263.66745 , -271.9075 , 364.11523],
			[-258.98856 , -213.80507 , 371.35303],
			[-251.81235 , -154.74184 , 375.63562],
			[-242.69043 , -95.2009 , 376.7576],
			[-232.27316 , -35.57102 , 374.69257],
			[-221.25783 , 23.84044 , 369.57257],
			[-210.33664 , 82.794754 , 361.667],
			[-200.1449 , 141.10892 , 351.3619],
			[-191.20917 , 198.64287 , 339.13916],
			[-183.89526 , 255.28653 , 325.5555],
			[-178.35654 , 310.94702 , 311.2219],
			[-174.48184 , 365.53583 , 296.78253],
			[-171.84369 , 418.95596 , 282.89398],
			[-169.77026 , 471.09668 , 270.2088],
			[-167.4172 , 521.829 , 259.35858],
			[-163.8397 , 570.9999 , 250.93733],
			[-158.0643 , 618.42804 , 245.48494],
			[-149.16092 , 663.8982 , 243.47037],
			[-136.31476 , 707.15656 , 245.27512],
			[-118.89818 , 747.9054 , 251.17648],
			[-96.54272 , 785.7982 , 261.3309],
			[-69.12683 , 820.447 , 275.75504]
		])


def load_pts2_JSpline5():
	return np.array(
		[
			[697.881 , -186.39595 , 95.001495],
			[703.5759 , -184.98099 , 102.725204],
			[708.8482 , -184.36266 , 111.216415],
			[713.67896 , -184.60008 , 120.3697],
			[718.04443 , -185.74562 , 130.06683],
			[721.916 , -187.84473 , 140.17947],
			[725.2611 , -190.93567 , 150.57175],
			[728.0436 , -195.04935 , 161.10281],
			[730.2243 , -200.20914 , 171.62958],
			[731.7617 , -206.43057 , 182.0092],
			[732.61237 , -213.72122 , 192.10178],
			[732.7311 , -222.0803 , 201.77295],
			[732.0721 , -231.49873 , 210.89647],
			[730.58936 , -241.95868 , 219.35681],
			[728.2367 , -253.43349 , 227.05188],
			[724.96893 , -265.88736 , 233.89548],
			[720.742 , -279.2752 , 239.82004],
			[715.5137 , -293.54242 , 244.77922],
			[709.244 , -308.62466 , 248.7504],
			[701.89594 , -324.4489 , 251.73271],
			[693.4358 , -340.9346 , 253.74484],
			[683.834 , -357.99463 , 254.82286],
			[673.06555 , -375.5365 , 255.01825],
			[661.1104 , -393.46326 , 254.39557],
			[647.9544 , -411.67487 , 253.0306],
			[633.5896 , -430.06897 , 251.00792],
			[618.01465 , -448.54218 , 248.419],
			[601.23596 , -466.99112 , 245.36005],
			[583.2676 , -485.31335 , 241.92976],
			[564.1322 , -503.40857 , 238.22736],
			[543.8615 , -521.1798 , 234.35037],
			[522.4968 , -538.5344 , 230.39255],
			[500.08972 , -555.3848 , 226.4417],
			[476.7025 , -571.65027 , 222.5776],
			[452.40875 , -587.2573 , 218.86996],
			[427.29205 , -602.1397 , 215.37881],
			[401.4448 , -616.2376 , 212.15552],
			[374.96667 , -629.4973 , 209.24313],
			[347.96353 , -641.8707 , 206.67726],
			[320.54593 , -653.31506 , 204.48657],
			[292.82797 , -663.7917 , 202.6936],
			[264.9259 , -673.2666 , 201.31528],
			[236.95686 , -681.7091 , 200.36368],
			[209.03752 , -689.09204 , 199.8466],
			[181.28282 , -695.3907 , 199.76831],
			[153.80464 , -700.5825 , 200.1301],
			[126.710526 , -704.64685 , 200.93106],
			[100.10234 , -707.56433 , 202.16867],
			[74.075005 , -709.3161 , 203.83945],
			[48.715126 , -709.8838 , 205.9396],
			[24.099747 , -709.2487 , 208.4658],
			[0.29682064 , -707.3926 , 211.4146],
			[-22.634336 , -704.2977 , 214.78206],
			[-44.642563 , -699.948 , 218.56335],
			[-65.68435 , -694.3287 , 222.75232],
			[-85.72339 , -687.4275 , 227.34106],
			[-104.730095 , -679.23474 , 232.31943],
			[-122.6811 , -669.74426 , 237.67473],
			[-139.55879 , -658.9531 , 243.39114],
			[-155.35086 , -646.8629 , 249.44942],
			[-170.04979 , -633.4799 , 255.82642],
			[-183.65237 , -618.8153 , 262.4947],
			[-196.15924 , -602.88635 , 269.42194],
			[-207.57443 , -585.71594 , 276.57074],
			[-217.9048 , -567.334 , 283.89813],
			[-227.15973 , -547.7774 , 291.35492],
			[-235.35043 , -527.0908 , 298.88568],
			[-242.49036 , -505.32547 , 306.42944],
			[-248.5954 , -482.53864 , 313.92114],
			[-253.68425 , -458.79254 , 321.2923],
			[-257.77847 , -434.15387 , 328.4727],
			[-260.903 , -408.69244 , 335.3909],
			[-263.08627 , -382.48083 , 341.97583],
			[-264.3604 , -355.59323 , 348.15753],
			[-264.76172 , -328.10468 , 353.86853],
			[-264.33075 , -300.0902 , 359.0447],
			[-263.1127 , -271.62415 , 363.6266],
			[-261.1575 , -242.77904 , 367.56036],
			[-258.5203 , -213.62494 , 370.79892],
			[-255.2616 , -184.22849 , 373.30322],
			[-251.44754 , -154.65227 , 375.043],
			[-247.15012 , -124.95366 , 375.99823],
			[-242.44754 , -95.18422 , 376.16],
			[-237.42262 , -65.39 , 375.53003],
			[-232.16103 , -35.61197 , 374.11978],
			[-226.74976 , -5.8863983 , 371.95007],
			[-221.27548 , 23.754726 , 369.05026],
			[-215.82297 , 53.283302 , 365.45773],
			[-210.47342 , 82.67473 , 361.2171],
			[-205.30281 , 111.90745 , 356.37955],
			[-200.3804 , 140.96262 , 351.00235],
			[-195.76697 , 169.8237 , 345.14807],
			[-191.51324 , 198.47594 , 338.88397],
			[-187.65836 , 226.90616 , 332.28137],
			[-184.2281 , 255.10217 , 325.4149],
			[-181.23338 , 283.05255 , 318.36197],
			[-178.66855 , 310.7461 , 311.2021],
			[-176.50983 , 338.17142 , 304.0161],
			[-174.71368 , 365.3167 , 296.8858],
			[-173.21906 , 392.16937 , 289.89307],
			[-171.94957 , 418.71606 , 283.11963],
			[-170.81589 , 444.94232 , 276.64636],
			[-169.71783 , 470.83258 , 270.55286],
			[-168.54672 , 496.37003 , 264.91693],
			[-167.18762 , 521.53625 , 259.81403],
			[-165.52151 , 546.31116 , 255.31664],
			[-163.42761 , 570.67316 , 251.49399],
			[-160.78558 , 594.5983 , 248.41135],
			[-157.47781 , 618.0609 , 246.12953],
			[-153.39165 , 641.03284 , 244.70444],
			[-148.42165 , 663.48346 , 244.18648],
			[-142.4718 , 685.3798 , 244.62012],
			[-135.4578 , 706.686 , 246.04324],
			[-127.3093 , 727.3633 , 248.48672],
			[-117.97214 , 747.3699 , 251.97394],
			[-107.40797 , 766.6612 , 256.52002],
			[-95.59389 , 785.18994 , 262.13147],
			[-82.52203 , 802.9066 , 268.80542],
			[-68.19923 , 819.7594 , 276.529],
			[-52.646576 , 835.69446 , 285.279],
			[-35.899094 , 850.6563 , 295.0209],
			[-18.005322 , 864.5881 , 305.70874],
			[0.97304964 , 877.4313 , 317.28412],
			[20.961565 , 889.1267 , 329.67575],
			[41.873386 , 899.6139 , 342.79898],
			[63.609673 , 908.8323 , 356.55505],
			[86.05996 , 916.7202 , 370.8305],
			[109.10254 , 923.21625 , 385.49677],
			[132.60486 , 928.259 , 400.40936],
			[156.42386 , 931.7871 , 415.40735],
			[180.40642 , 933.73987 , 430.31287],
			[204.3916 , 934.0581 , 444.935],
			[228.2129 , 932.68616 , 459.0735],
			[251.70059 , 929.57275 , 472.52286],
			[274.68402 , 924.67255 , 485.0763],
			[296.9938 , 917.94763 , 496.52954],
			[318.4642 , 909.3688 , 506.68463],
			[338.93536 , 898.91675 , 515.3541],
			[358.25568 , 886.5838 , 522.36475],
			[376.2838 , 872.375 , 527.5614],
			[392.8914 , 856.30963 , 530.81116],
			[407.96497 , 838.4225 , 532.00684],
			[421.40845 , 818.76544 , 531.0714],
			[433.1453 , 797.40845 , 527.96155],
			[443.1209 , 774.44135 , 522.6716],
			[451.3048 , 749.9749 , 515.23755],
			[457.69293 , 724.14246 , 505.74088],
			[462.30573 , 697.09656 , 494.30377],
			[465.18567 , 669.00586 , 481.0838],
			[466.3953 , 640.05225 , 466.26935],
			[466.01526 , 610.42706 , 450.0744],
			[464.14172 , 580.32886 , 432.73355],
			[460.88492 , 549.9592 , 414.49728],
			[456.36627 , 519.5206 , 395.62686],
			[450.71692 , 489.21265 , 376.38922],
			[444.07513 , 459.229 , 357.0522],
			[436.58435 , 429.75436 , 337.87958],
			[428.3911 , 400.96136 , 319.12598],
			[419.64285 , 373.0073 , 301.03207],
			[410.48584 , 346.03094 , 283.81937],
			[401.06293 , 320.1496 , 267.68552],
			[391.51154 , 295.45587 , 252.79922],
			[381.96146 , 272.01437 , 239.29523],
			[372.5345 , 249.86414 , 227.27643],
			[363.344 , 229.02074 , 216.81598],
			[354.49448 , 209.47852 , 207.95927],
			[346.0811 , 191.21277 , 200.72595],
			[338.18936 , 174.18195 , 195.11212],
			[330.8946 , 158.3299 , 191.09225],
			[324.2616 , 143.58806 , 188.62126],
			[318.34427 , 129.87766 , 187.63663],
			[313.185 , 117.11194 , 188.06046],
			[308.81445 , 105.19833 , 189.80136],
			[305.251 , 94.04071 , 192.75673],
			[302.50037 , 83.54156 , 196.81473],
			[300.5552 , 73.60422 , 201.85623],
			[299.39465 , 64.13504 , 207.75705],
			[298.98392 , 55.045666 , 214.3898],
			[299.2739 , 46.255157 , 221.62616],
			[300.2033 , 37.689915 , 229.33755],
			[301.701 , 29.283482 , 237.39603],
			[303.68796 , 20.976437 , 245.67511],
			[306.07977 , 12.716217 , 254.05069],
			[308.78857 , 4.4569798 , 262.40173],
			[311.7255 , -3.8405418 , 270.61133],
			[314.8026 , -12.209188 , 278.5672],
			[317.9354 , -20.675522 , 286.1629],
			[321.04474 , -29.259977 , 293.2985],
			[324.05923 , -37.977 , 299.88123],
			[326.91736 , -46.835205 , 305.82672],
			[329.5697 , -55.837517 , 311.05957],
			[331.98108 , -64.98132 , 315.51416],
			[334.13287 , -74.258606 , 319.13565],
			[336.0252 , -83.65612 , 321.8807],
			[337.67902 , -93.1555 , 323.7184],
			[339.13354 , -102.733765 , 324.62946],
			[340.4438 , -112.36373 , 324.60544],
			[341.6778 , -122.01449 , 323.64786],
			[342.91406 , -131.65187 , 321.76773],
			[344.23907 , -141.23889 , 318.98456],
			[345.7445 , -150.7362 , 315.32562],
			[347.5247 , -160.1026 , 310.82532],
			[349.674 , -169.29536 , 305.52414],
			[352.28442 , -178.27086 , 299.46826],
			[355.4426 , -186.9849 , 292.70853],
			[359.22754 , -195.39319 , 285.29974],
			[363.70782 , -203.4519 , 277.29987],
			[368.9391 , -211.11804 , 268.76944],
			[374.96143 , -218.34985 , 259.77057],
			[381.79678 , -225.10738 , 250.36632],
			[389.44626 , -231.3529 , 240.6199],
			[397.8922 , -237.0513 , 230.5946],
			[407.1005 , -242.1705 , 220.35394],
			[417.0221 , -246.6819 , 209.96146],
			[427.59567 , -250.56068 , 199.4808],
			[438.74933 , -253.78635 , 188.97571],
			[450.40265 , -256.34302 , 178.50996],
			[462.46902 , -258.21997 , 168.14737],
			[474.8573 , -259.41196 , 157.95178],
			[487.4742 , -259.91953 , 147.98706],
			[500.2263 , -259.74963 , 138.31708],
			[513.02185 , -258.9159 , 129.00566],
			[525.7732 , -257.43912 , 120.1166],
			[538.3986 , -255.34753 , 111.71365],
			[550.8243 , -252.6774 , 103.86048],
			[562.9866 , -249.47333 , 96.620705],
			[574.83405 , -245.78862 , 90.057816],
			[586.3261 , -241.68446 , 84.23411],
			[597.43225 , -237.22888 , 79.20952],
			[608.13043 , -232.49582 , 75.04056],
			[618.40625 , -227.5643 , 71.77915],
			[628.2518 , -222.5173 , 69.47156],
			[637.6643 , -217.44101 , 68.15721],
			[646.64514 , -212.4237 , 67.86763],
			[655.19885 , -207.55495 , 68.62529],
			[663.3316 , -202.92459 , 70.442505],
			[671.05023 , -198.62187 , 73.32034],
			[678.3611 , -194.73436 , 77.247444],
			[685.26917 , -191.34718 , 82.198944],
			[691.77637 , -188.54198 , 88.135376]
		])


def Rodrigues(v: np.ndarray, k: np.ndarray, theta: float):
	""" rotate vector v about axis k (unit vector) """
	k /= np.linalg.norm(k)

	c = np.cos(theta)
	s = np.sin(theta)
	v_rot = v*c + np.cross(k,v)*s + k*(k.dot(v))*(1-c)
	return v_rot


def test_Rodrigues():
	""" """
	# z points up
	v = np.array([-2.,0,1.])
	k = np.array([0,0,1.])
	theta = np.deg2rad(90 + 45)
	v_rot = Rodrigues(v,k,theta)
	# array([ 1.41421356, -1.41421356,  1. ])
	pdb.set_trace()
	
	# z points up, but we stay in the plane
	v = np.array([2.,0,0.])
	k = np.array([1,1,0.])
	theta = np.deg2rad(180)
	pdb.set_trace()
	v_rot = Rodrigues(v,k,theta)
	# array([-0.,  2., -0.])




# def Rodrigues_vectorized(v: np.ndarray, k: np.ndarray, theta: float):
# 	""" """
# 	R = 

def get_ngon_points(o, i, j, n: int, radius: float):
	""" """
	dim = 3

	if n == 6:
		angle = np.deg2rad(360 / n)
		hex_coeffs = np.array(
			[
				[ 1				, 0				 ],
				[ np.cos(angle)	, np.sin(angle)	 ],
				[ -np.cos(angle), np.sin(angle)	 ],
				[ -1			, 0				 ],
				[ -np.cos(angle), -np.sin(angle) ],
				[ np.cos(angle), -np.sin(angle)	 ]
			])
		pts = np.zeros((6,dim))
		for idx, (a,b) in enumerate(hex_coeffs):
			pts[idx] = o + (i * a) + (j * b)

	else:
		raise RuntimeError

	return pts


def test_get_ngon_points():
	""" """
	o = np.zeros(2)
	i = np.array([1,0])
	j = np.array([0,1])
	pts = get_ngon_points(o, i, j, n=6, radius=1)

	pts = np.vstack([pts,pts[0]])

	plt.scatter(pts[:,0],pts[:,1],10,color='r')
	plt.plot(pts[:,0],pts[:,1],color='g')
	plt.axis('equal')
	plt.show()


def plot_ngon_mayavi():
	""" """
	o = np.array([1,1,1])
	i = np.array([1,-1,1])
	j = np.array([-1,1,1])

	# o = np.array([0,0,0])
	# i = np.array([1,0,0])
	# j = np.array([0,1,0])

	points = get_ngon_points(o, i, j, n=6, radius=1)
	
	fig = mayavi.mlab.figure(figure=None, bgcolor=(1, 1, 1), fgcolor=None, engine=None, size=(1600, 1000))
	draw_mayavi_line_segment(
	    fig,
	    np.vstack([points,points[0]])
	)
	fig = draw_coordinate_frame_at_origin(fig)
	#l = mayavi.mlab.plot3d(x, y, z, np.sin(mu), tube_radius=0.025, colormap="Spectral")
	mayavi.mlab.show()


if __name__ == '__main__':
	#draw_JSpline()
	#test_twistFreeQuadMesh1()
	#test_twistFreeQuadMesh2()
	test_twistFreeQuadMesh3()
	#test_Rodrigues()

	#test_get_angle_between()
	#test_get_ngon_points()

	#plot_ngon_mayavi()
```

## References
1. https://www.microsoft.com/en-us/research/wp-content/uploads/2016/12/Computation-of-rotation-minimizing-frames.pdf, implemented [here](https://math.stackexchange.com/questions/2843307/getting-consistent-normals-along-a-3d-bezier-curve)
