---
layout: post
title:  "Lie Groups and Rigid Body Kinematics"
permalink: /lie-groups/
excerpt: "SO(2), SO(3), SE(2), SE(3), Lie algebras"
mathjax: true
date:   2018-12-28 11:00:00
mathjax: true

---
Table of Contents:
- [Why do we need Lie Groups?](#whyliegroups)
- [Lie Groups](#liegroups)
- [SO(N)](#son)
- [SO(2)](#so2)
- [SO(3)](#so3)
- [SE(2)](#se2)
- [SE(3)](#se3)
- [Conjugation in Group Theory](#conjugation)
- [The Lie Algebra](#lie-algebra)

<a name='whyliegroups'></a>

## Why do we need Lie Groups?

Rigid bodies have a state which consists of position and orientation. When sensors are placed on a rigid body (e.g. a robot), they provide measurements in the body frame. Suppose we wish to take a measurement $$y_b$$ from the body frame and move it to the world frame, $$y_w$$. We can do this via left multiplication with a transformation matrix $${}^{w}T_{b}$$, a member of the matrix Lie groups, that transports the point from one space to another space:

$$
y_w = {}^{w}T_{b} y_b
$$

## What is a *group*?
A group is a set $$G$$, with an operation of (binary) multiplication $$\circ$$ on elements of $$G$$ which is:
* closed: If $$g_1, g_2 \in G$$ then also $$g_1 \circ g_2 \in G$$;
* associative: $$(g_1 \circ g_2) \circ g_3 = g_1 \circ (g_2 \circ g_3)$$, for all $$g_1, g_2, g_3 \in G$$;
* unit element $$e$$: $$e \circ g = g \circ e = g$$, for all $$g \in G$$;
* invertible: For every element $$g \in G$$, there exists an element $$g^{−1} \in G$$ such that $$g \circ g^{−1} = g^{−1} \circ g = e$$.

More details can be found in [5].

<a name='liegroups'></a>

## Lie Groups

When we are working with pure rotations, we work with Special Orthogonal groups, $$SO(\cdot)$$. When we are working with a rotation and a translation together, we work with Special Euclidean groups $$SE(\cdot)$$.

Lie Groups are unique because they are **both a group and a manifold**. They are continuous manifolds in high-dimensional spaces, and have a group structure. I'll describe them in more detail below.

<a name='son'></a>

### SO(N)

Membership in the Special Orthogonal Group $$SO(N)$$ requires two matrix properties:
-	$$R^TR = I $$
- 	$$ \mbox{det}(R) = +1 $$

This gives us a very helpful property: $$R^{-1} = R^T$$, so the matrix inverse is as simple as taking the transpose.  We will generally work with $$SO(N)$$, where $$N=2,3$$, meaning the matrices are rotation matrices $$R \in \mathbf{R}^{2 \times 2}$$ or $$R \in \mathbf{R}^{3 \times 3}$$.

These rotation matrices $$R$$ are not commutative.

<a name='so2'></a>

### SO(2)

$$SO(2)$$ is a 1D manifold living in a 2D Euclidean space, e.g. moving around a circle.  We will be stuck with singularities if we use 2 numbers to parameterize it, which would mean kinematics break down at certain orientations.

$$SO(2)$$ is the space of orthogonal matrices that corresponds to rotations in the plane.

**A simple example**:
Let's move from the body frame $$b$$ to a target frame $$t$$:

$$
P_t = {}^tR_b(\theta) P_b
$$

$$
\begin{bmatrix}
5 \mbox{ cos} (\theta) \\
5 \mbox{ sin} (\theta)
\end{bmatrix} = \begin{bmatrix} cos(\theta) & -sin(\theta) \\ sin(\theta) & cos(\theta) \end{bmatrix} * \begin{bmatrix} 5 \\ 0 \end{bmatrix}
$$

As described in [3], another way to think of this is to consider that a robot can be rotated counterclockwise by some angle $$\theta \in [0,2 \pi)$$ by mapping every $$(x,y)$$ as:

$$
(x, y) \rightarrow (x \mbox{ cos } \theta − y \mbox{ sin } \theta, x \mbox{ sin } \theta + y \mbox{ cos } \theta).
$$

<a name='so3'></a>

### SO(3)

There are several well-known parameterizations of $$R \in SO(3)$$:
1. $$R \in \mathbf{R}^{3 \times 3}$$ full rotation matrix, 9 parameters -- there must be 6 constraints
2. Euler angles, e.g. $$(\phi, \theta, \psi)$$, so 3 parameters
3. Angle-Axis parameters $$(\vec{a}, \phi)$$, which is 4 parameters and 1 constraint (unit length)
4. Quaternions ($$q_0,q_1,q_2,q_3)$$, 4 parameters and 1 constraint (unit length)

There are only 3 degrees of freedom in describing a rotation. But this object doesn't live in 3D space. It is a 3D manifold, embedded in a 4-D Euclidean space.

Parameterizations 1,3,4 are overconstrained, meaning they employ more parameters than we strictly need. With overparameterized representations, we have to do extra work to make sure we satisfy the constraints of the representation.

As it turns out $$SO(3)$$ cannot be parameterized by only 3 parameters in a non-singular way.


**Euler Angles**
One parameterization of $$SO(3)$$ is to imagine three successive rotations around different axes. The Euler angles encapsulate yaw-pitch-roll: first, a rotation about the z-axis (yaw, $$\psi$$). Then, a rotation about the pitch axis by $$\theta$$ (via right-hand rule), and finally we perform a roll by $$\phi$$.

<div class="fig figcenter fighighlight">
  <img src="/assets/euler_angles.jpg" width="65%">
  <div class="figcaption">
    The Euler angles.
  </div>
</div>

The sequence of successive rotations is encapsulated in $${}^{w}R_b$$:

$$
{}^{w}R_b = R_R(\phi) R_P(\theta) R_Y (\psi)
$$

As outlined in [3], these successive rotations by $$(\phi, \theta, \psi)$$ are defined by:

$$
R_{yaw} = R_y (\psi) = \begin{bmatrix}
\mbox{cos} \psi & -\mbox{sin} \psi  & 0 \\
\mbox{sin} \psi & \mbox{cos} \psi & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

$$
R_{pitch} = R_p (\theta) = \begin{bmatrix}
\mbox{cos} \theta & 0 & \mbox{sin} \theta \\
 0 & 1 & 0 \\
 -\mbox{sin} \theta & 0 & \mbox{cos} \theta \\
\end{bmatrix}
$$

$$
R_{roll} = R_R (\phi) = \begin{bmatrix}
1 & 0 & 0 \\
0 & \mbox{cos} \phi & -\mbox{sin} \phi \\
0 & \mbox{sin} \phi &  \mbox{cos} \phi \\
\end{bmatrix}
$$

You have probably noticed that each rotation matrix $$\in \mathbf{R}^{3 \times 3}$$ above is a simple extension of the 2D rotation matrix from $$SO(2)$$. For example, the yaw matrix $$R_{yaw}$$ performs a 2D rotation with respect to the $$x$$ and $$y$$ coordinates while leaving the $$z$$ coordinate unchanged [3].

<a name='se2'></a>

### SE(2)

The real space $$SE(2)$$ are $$3 \times 3$$ matrices, moving a point in homogenous coordinates to a new frame. It is important to remember that this represents a rotation followed by a translation (not the other way around).  A rigid body which translates and rotates on a 2D plane has 3 DOF, e.g. a ground robot.

$$
T = \begin{bmatrix} x_w \\ y_w \\ 1 \end{bmatrix} = \begin{bmatrix}
 R_{2 \times 2}& & t_{2 \times 1}  \\
& \ddots & \vdots  \\
0 & 0 & 1
\end{bmatrix} * \begin{bmatrix} x_b \\ y_b \\ 1 \end{bmatrix} 
$$

By adding an extra dimension to the input points and transformation matrix $$T$$ the translational part of the transformation is absorbed [3].

<a name='se3'></a>

### SE(3)

The set of rigid body motions, or special Euclidean transformations, is a (Lie) group, the so-called special Euclidean group, typically denoted as SE(3). The real space $$SE(3)$$ is a 6-dimensional manifold. Its dimensions is exactly the number of degrees of freedom of a free-floating rigid body in space [3]. $$SE(3)$$ can be parameterized with a $$4 \times 4$$ matrix as follows:

$$
\begin{bmatrix}
& & & \\
& R_{3 \times 3} & &  t_{3 \times 1} \\
& & & \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

What is the inverse of an SE(3) object? Consider a transformation of a point in the body frame $$p_b$$ to a point in the world frame $$p_w$$. Both points $$p_b, p_w$$ must be in homogeneous coordinates. We can invert it as follows:

$$
\begin{aligned}
p_w &= {}^{w}T_{b} p_b \\
p_w &= \begin{bmatrix} {}^wR_b & | & {}^wt_b \end{bmatrix} p_b \\
p_w &=  {}^wR_b p_b + {}^wt_b \\
p_w - {}^wt_b &= {}^wR_b p_b \\
({}^wR_b)^{-1} (p_w - {}^wt_b) &= ({}^wR_b)^{-1} {}^wR_b p_b \\
({}^wR_b)^T (p_w - {}^wt_b) &= p_b \\
({}^wR_b)^T p_w - ({}^wR_b)^T {}^wt_b &= p_b \\
\begin{bmatrix} ({}^wR_b)^T & | & -({}^wR_b)^T {}^wt_b \end{bmatrix}p_w &= p_b \\
p_b &= \begin{bmatrix} ({}^wR_b)^T & | & -({}^wR_b)^T{}^wt_b \end{bmatrix}p_w \\
p_b &= {}^{w}T_{b}^{-1} p_w
\end{aligned}
$$

Thus if $$T = \begin{bmatrix}R & t \\ 0 & 1\end{bmatrix}$$, then $$T^{-1} = \begin{bmatrix} R^T & -R^Tt \\ 0 & 1 \end{bmatrix}$$. You can find my simple implementation of an object-oriented SE(3) class [in Python here](https://github.com/argoai/argoverse-api/blob/master/argoverse/utils/se3.py#L47).

<a name='conjugation'></a>

## Conjugation in Group Theory

Surprisingly, movement in $$SE(2)$$ can always be achieved by moving somewhere, making a rotation there, and then moving back. 

$$
SE(2) = (\cdot) SO(2) (\cdot)
$$

If we move to a point by vector movement $$p$$, essentially we perform: $$B-p$$, then go back with $$p + \dots $$.

$$
B^{\prime} = \begin{bmatrix}
R & t \\
0 & 1
\end{bmatrix}_B = \begin{bmatrix}
I & p \\
0 & 1
\end{bmatrix} \begin{bmatrix}
R & 0 \\
0 & 1
\end{bmatrix} \begin{bmatrix}
I & -p \\
0 & 1
\end{bmatrix}_B
$$


<a name='lie-algebra'></a>

## Connecting Spatial Coordinates and Spatial Velocities: The Lie Algebra

The Lie Algebra maps spatial coordinates to spatial velocity.

In the case of $$SO(3)$$, the Lie Algebra is the space of skew-symmetric matrices



Rigid Body Kinematics: we want a differential equation (ODE) that links $$\vec{\omega}$$ and $${}^{^w}R_b$$


in particular,
$$  {}^{^w} \dot{R}_r = f({}^{^w}R_b, \vec{\omega}) $$

$$
\begin{aligned}
\frac{ \partial }{\partial t}(R^R = I)
 \\
\dot{R}^TR + R^T \dot{R} = 0 \\
\dot{R}^TR = -R^T \dot{R} 
\end{aligned}
$$

we define $$\Omega = R^T \dot{R} $$

$$
\Omega^T = (R^T \dot{R} )^T = \dot{R}^TR  = -R^T \dot{R}  = -\Omega
$$

Skew-symmetric! $$\Omega^T = -\Omega$$

In fact, 

$$
\Omega = \begin{bmatrix}
0 & -\omega_z & \omega_y \\
\omega_z & 0 & -\omega_x \\
-\omega_y & \omega_x & 0
\end{bmatrix}
$$

where $$\vec{\omega} = \begin{bmatrix} \omega_x \\ \omega_y \\ \omega_z \end{bmatrix}$$ is the angular velocity

Notation! the "hat" map

$$
\bar{\omega}^{\hat{}} =\Omega 
$$

the "v" map

$$
\Omega^{v} = \bar{\omega} 
$$


So we have

$$
\begin{aligned}
\dot{R} = R \Omega \\
\Omega = \bar{\omega}^{\hat{}}
\end{aligned}
$$

In terms of frames, we have Poisson's kinematic equation.

The intrinsic equation is (where $$\vec{\omega}_b$$ in the body frame is $$\Omega_b$$):

$$
{}^{^w}\dot{R}_b =  {}^{^w}R_b \Omega_b
$$


The "strap-down" equation (extrinsic)

$$
{}^{^b}\dot{R}_w =  - \Omega_b {}^{^b}R_w
$$

Take vector on aircraft, put it into the coordinate frame of the world


$$
{}_{^w}R_b = \begin{bmatrix} | & | & | \\ {}^{^w} \hat{x}_b & {}^{^w}\hat{y}_b & {}^{^w} \hat{z}_b \\ | & | & |  \end{bmatrix}
$$


$$
\Omega \vec{v}_b = \omega \times \vec{v}_b
$$

Can we use this for filtering? Can we use $$\dot{R} = R \Omega $$ directly in an EKF, UKF
We'd have to turn it into discrete-time 
Naively, you might do the first order Euler approximation:

$$
\frac{ R_{t + \delta_t} -R_t}{\delta t} \approx R_t \Omega_t
$$

This does not work!

You won't maintain orthogonality! The defining property of $$SO(3)$$ was orthogonality, so

$$
R_{t + \delta t} \not\in SO(3)
$$

and pretty soon

$$
\begin{aligned}
R_{t + \delta t}^T R_{t + \delta t} \neq I \\
\mbox{det }(R_{t+\delta t}) \neq +1
\end{aligned}
$$

The 3x3 matrix will not be recognizable of a rotation

## The Exponential Map

The matrix exponential 

$$
e^{ \hat{\omega}t} = I + \hat{\omega}t + \frac{ (\hat{\omega}t)^2}{2!} + \cdots + \frac{ (\hat{\omega}t)^n}{n!} + \cdots
$$

defines a map from the space $$so(3)$$ to $$SO(3)$$, which we often call the *exponential map* [5].

For any rotation matrix $$R \in SO(3)$$, there exists a $$\omega \in \mathbf{R}^3, \|\omega\|=1$$ and $$t \in \mathbf{R}$$ such that $$R = e^{ \hat{\omega}t}$$. This theorem is quite powerful: it means that any rotation matrix can be realized by rotating around some fixed axis by a certain angle. This map is not one-to-one.

## Twists

A $$4 \times 4$$ matrix of the form $$\hat{\xi}$$ is called a twist. The set of all twists is denoted as $$se(3)$$ [4,5]:

$$
se(3) = \Bigg\{ \hat{\xi} = \begin{bmatrix} \hat{\omega} & v \\ 0 & 0 \end{bmatrix} \mid \hat{\xi} \in so(3), v \in \mathbf{R}^3 \Bigg\} \subset \mathbf{R}^{4 \times 4}
$$

$$se(3)$$ is called the tangent space (or Lie algebra) of the matrix group $$SE(3)$$.

Why do we care about twists? It turns out that a rigid body can be moved from one position to any other by a movement consisting of (1) a rotation about a straight line (2) followed by a translation parallel to that line. This type of motion is *screw motion*, and its infinitesimal version is called a *twist*. The beauty of a twist is that it describes the instantaneous velocity of a rigid body in terms of its **linear and angular components** [5]. It is the matrix exponential that maps a twist into its corresponding screw motion.

## Conclusion

A *holonomic robot* is one which is able to move instantaneously in
any direction in the space of its degrees of freedom. Otherwise a robot is called *non-holonomic*.


## References

[1] Frank Dellaert. Lecture Presentations of MM 8803: Mobile Manipulation, taught at the Georgia Institute of Technology, Fall 2018.

[2] Mac Schwager. Lecture Presentations of AA 273: State Estimation and Filtering for Aerospace Systems, taught at Stanford University in April-June 2018. 

[3] Steven M. LaValle. *Planning Algorithms*. Cambridge University Press, 2006, New York, NY, USA. 

[4] Murray, R.M., Li, Z., Sastry, S.S., Sastry, S.S.: A mathematical introduction to robotic manipulation. CRC press (1994). [PDF](https://www.cds.caltech.edu/~murray/books/MLS/pdf/mls94-complete.pdf).

[5] Ma, Yi and Soatto, Stefano and Kosecka, Jana and Sastry, S. Shankar: An Invitation to 3-D Vision: From Images to Geometric Models. Springer Verlag (2003). [PDF](https://www.eecis.udel.edu/~cer/arv/readings/old_mkss.pdf).

[6] Murray et al. http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.671.7040&rep=rep1&type=pdf


## Manifolds: SO(2)

SO(2), 2x2 matrices, with determinant one, orthogonal, 1 dimensional manifold in a 4 dimensional space. (c,s) in R^2, lie algebra is the tangent space at the origin. Tangent Bundle. Can get tangent space at a single point R. every tangent space is a rotated copy of that lie algebra


## Manifolds: 3d Lines

(R, p) 
R has 9 numbers, 3d rotation
p is 2d in xy plane
embedding space is R 11

but 4d manifold

Tangent space is TL_p, isomorphic to R4. THERE IS no lie algebra since this is not a lie group.

exponential map takes us from tangent space to the manifold


retract for any manifold takes us from tangent space to the manifold


**Rigid Body Kinematics and Filtering**

Kinematics with Euler Angles



$$
\dot{R} = f(R, \vec{w}), (\dot{\phi}, \dot{\theta}, \dot{\psi}) = f(\phi, \theta, \psi, \vec{\omega})
$$



$$
\vec{\omega} = \begin{bmatrix} w_x \\ w_y \\ w_z \end{bmatrix} = \begin{bmatrix} 1 & 0 & -\mbox{sin} \theta \\ 0 & \mbox{cos} \phi & \mbox{sin} \phi \mbox{cos} \phi \\ 0 & -\mbox{sin} \phi & \mbox{cos} \phi \mbox{cos} \theta \end{bmatrix} \begin{bmatrix} \dot{\phi} \\ \dot{\theta} \\ \dot{\psi} \end{bmatrix}
$$

We want to invert this matrix and solve for $$\begin{bmatrix} \dot{\phi} \\ \dot{\theta} \\ \dot{\psi} \end{bmatrix}$$



## As an aside: Quaternions


```python
import numpy as np

# import quaternion


EPSILON = 1e-10

# make sure quaternions are normalized (to be unit quaternions)
# unit quaternions are a way to compactly represent 3D rotations
# while avoiding singularities or discontinuities (e.g. gimbal lock).


def quat_mult(q1, q2):
    """
	Multiply two quaternions.

		Args:
		-	q1: NumPy n-d array, with dim (4 x 1)
		-	q2: NumPy n-d array, with dim (4 x 1)

	We recall that a cross product (u x v) is defined as 
		(u_2 * v_3) - (u_3 * v_2)
		(u_3 * v_1) - (u_1 * v_3)
		(u_1 * v_2) - (u_2 * v_1)
	"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    # w = (w_1 * w_2) - < v_1, v_2 >
    w = (w1 * w2) - (x1 * x2) - (y1 * y2) - (z1 * z2)

    # v = w_1 * v_2 + w_2 * v_1 + (v_1 x v_2)
    x = (w1 * x2) + (x1 * w2) + y1 * z2 - z1 * y2
    y = (w1 * y2) + (y1 * w2) + z1 * x2 - x1 * z2
    z = (w1 * z2) + (z1 * w2) + x1 * y2 - y1 * x2
    return np.array([w, x, y, z])


def quat_conjugate(q):
    """
		Args:
		-	q: NumPy n-d array, with dim (4 x 1)

	The conjugate of a quaternion is the same as the inverse, as long as the quaternion is unit-length
	"""
    w, x, y, z = q
    return (w, -x, -y, -z)


def quat_vec_mult(q1, v1):
    """ 
	quaternion-vector multiplication 
	Apply a quaternion-rotation to a vector.
		v' = q * v * q_conj

		Args:
		-	q1: 
		-	v1:

		Returns:
		-	
	"""
    q2 = (0.0,) + v1
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]


def quat_normalize(q):
    """
	Normalize a quaternion.
		Args:
		-	q: NumPy n-d array, with dim (4 x 1) representing
				(w,x,y,z)
	"""
    return q / np.linalg.norm(q)


def quat2rotmat(q):
    """
	Convert a quaternion into a matrix.

		Args:
		-	q: NumPy n-d array, with dim (4 x 1), representing
				(w,x,y,z)

		Returns:
		-	R: Numpy array of shape (3,3)
	"""
    if (np.linalg.norm(q) - 1.0) > EPSILON:
        q /= np.linalg.norm(q)
        # q = np.multiply(q, np.tile(1./np.linalg.norm(q),4))
    w, x, y, z = q

    x2 = x * x
    y2 = y * y
    z2 = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array(
        [
            [1.0 - 2.0 * (y2 + z2), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (x2 + z2), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (x2 + y2)],
        ]
    )


def quat_normalization_unit_test(q_quat, q_arr):

    # print( quaternion.quaternion_normalized(q_quat) )

    print(quat_normalize(q_arr))


def conjugate_unit_test(q_quat, q_arr):
    """ Test Conjugate functionality """
    print(q_quat.conjugate())
    print(quat_conjugate(q_arr))


def quat_2_rot_mat_unit_test(q_quat, q_arr):
    """ Test quaternion-vector product functionality """
    # print( quaternion.as_rotation_matrix(q_quat) )

    print(quat2rotmat(q_arr))


def unit_tests():
    """ Check against the Numpy Quaternion add-on package """
    q_quat = np.quaternion(4, 3, 2, 1)
    q_arr = np.array([4, 3, 2, 1])

    # q_quat = np.quaternion(	0.73029674, 0.54772256, 0.36514837, 0.18257419)
    # q_arr = np.array([		0.73029674, 0.54772256, 0.36514837, 0.18257419])

    # quat_normalization_unit_test(q_quat, q_arr)

    # conjugate_unit_test(q_quat, q_arr)
    quat_2_rot_mat_unit_test(q_quat, q_arr)


if __name__ == "__main__":
    unit_tests()
```




