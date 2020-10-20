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


