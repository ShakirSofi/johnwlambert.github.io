---
layout: post
title:  "Pose Graph SLAM"
permalink: /pose-slam/
excerpt: "Simultaneous Localization and Mapping, GraphSLAM, loop closures"
mathjax: true
date:   2018-12-27 11:00:00
mathjax: true

---
Table of Contents:
- [Gauss-Newton Optimization](#sfmpipeline)
- [2D SLAM: Pose-Pose Constraints](#2d-pose-pose)
- [2D SLAM: Pose-Landmark Constraints ](#2d-pose-landmark)

<a name='sfmpipeline'></a>

## Pose Graph SLAM

Pose Graph SLAM is one of the most important tools in mobile robotics. Informally, the problem is to build a map of your environment nad simultaneously localize yourself within this map. 

Numerical linear algebraic techniques lie at the core of solutions to the SLAM problem. We will demonstrate the strengths and weaknesses of Levenberg-Marquardt optimization, and demonstrate how to exploit sparsity in the solution of associated least-squares problems. Specifically, we compare naive direct methods with a Cholesky factorization that avoids ``fill-in'' of sparse matrices. We plan to evaluate our solutions quantitatively (in terms of least square residuals) and qualitatively, via the visual quality of the constructed map on two small datasets.



Simultaneous Localization and Mapping (SLAM) is the problem of estimating an observers' position from local sensor measurements only, while creating a consistent map of the environment.  It is one of the central problems in robotics and maps generate by SLAM are considered by many to be a prerequisite for creating safe autonomous vehicles. A common representation of a ``map'' is a set of ``landmarks'' or ``features'', e.g. 3D points. Pose in 6 degrees of freedom (DOF) can be expressed as a member of SE(3), or in 3 degrees of freedom, SE(2).  Integrating noisy local sensor data into a consistent map is difficult because the accumulation of small local errors leads to larger global inconsistencies. \cite{kaess:08:thesis}. In fact, relying upon odometry and point cloud scan matching measurements alone will lead to sufficient accumulation error to render any localization estimate to be completely incorrect. However, by establishing loop closures, we can correct the entire map to a more accurate representation of the environment [2].

PoseGraph SLAM is a particular flavor of SLAM that is particularly amenable to working with laser rangefinders (LiDAR). These sensors produce point clouds at successive timestamps that can be registered with the ICP algorithm, yielding relative pose measurements [2]. These relative pose measurements form constraint that connect the poses of the robot while it is moving. Due to sensor noise, constraints are inherently uncertain. If a robot revisits previously-seen part of the environment, we can generate constraints between non-successive poses (``loop closures''). We can use a graph to represent the problem, and every node in the graph corresponds to a pose of the robot during mapping. Every edge between two nodes corresponds to a spatial constraint between them. In graph-based SLAM, we build the graph and find a node configuration that minimizes the error introduced by the constraints. Once we have the graph, we determine the most likely map by correcting the nodes. The problem formulation lends itself to minimizing the sum of the squared errors in  in an overdetermined system of equations. 


We can model the maximum likelihood estimate as a  least-squares problem, with nonlinear terms.


$$
F(x) = \sum\limits_{(i,j) \in \mathcal{C}}
$$
















## 2D-SLAM

\subsection{PoseGraph SLAM With Certainty Estimates: Incorporating the Information Matrix}

Constrain in the graph from a measurement $z_t^i$:
\begin{equation}
    (z_t^i - h(x_t, m_j))^T Q_t^{-1}  (z_t^i - h(x_t, m_j))
\end{equation}

Constraint in the graph from robot motion, with control input $u_t$:
\begin{equation}
    (x_t - g(u_t,x_{t-1}))^T R_t^{-1} (x_t - g(u_t,x_{t-1}))
\end{equation}

Sum of all constraints in the graph will be:
\begin{equation}
    \begin{aligned}
    J_{GraphSLAM} &= x_0^T \Omega_0 x_0 + \sum\limits_t  (x_t - g(u_t,x_{t-1}))^T R_t^{-1} (x_t - g(u_t,x_{t-1})) \\
    & + \sum\limits_t \sum\limits_i (z_t^i - h(x_t, m_j))^T Q_t^{-1}  (z_t^i - h(x_t, m_j))
    \end{aligned}
\end{equation}
It is a function defined over pose variables $x_{1:t}$ and all feature locations in the map $m$. We include an anchoring constraint $x_0^T \Omega_0 x_0$, which anchoirs the absolute coordinates of the map by initializing the very first pose of the robot as $(0,0,0)^T$ \cite{thrun:2005:probrobotics}.

If the robot moves from $x_i$ to $x_{i+1}$, the edge corresponds to odometry. We construct a virtual sensor measurement about the position of $x_j$ seen from $x_i$.





Using homogeneous coordinates for our points and a transformation 
\begin{equation}
\begin{array}{lll}
    T=\begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix} \in SE(3), & R \in SO(3), & t \in \mathbbm{R}^3
    \end{array}
\end{equation}
Odometry based edge:
\begin{equation}
    X_i^{-1}X_{i+1}
\end{equation}

Observation based edge:
\begin{equation}
    X_i^{-1}X_j
\end{equation}

The Information matrix $\Omega_{ij}$ for each edge encodes its uncertainty. The bigger the $\Omega_{ij}$, the more the edge ``matters'' in the optimization.Goal
\begin{equation}
\begin{aligned}
    x^{\star} = \mbox{arg } \underset{x}{\mbox{min}} \sum\limits_i \sum\limits_j e_{ij}^T \Omega_{ij} e_{ij} \\
     x^{\star} = \mbox{arg } \underset{x}{\mbox{min}} \sum\limits_i \sum\limits_j e_{ij}^T(x_i,x_j) \Omega_{ij} e_{ij}(x_i,x_j)
\end{aligned}
\end{equation}
In our case, the state vector $x$ resembles
\begin{equation}
    x^T = \begin{bmatrix} x_1^T & x_2^T & \cdots & x_n^T \end{bmatrix}
\end{equation}
where each block represents one node in the graph. Here,
\begin{equation}
    \vec{e}_{ij} = Z_{ij}^{-1}(X_i^{-1}X_j)
\end{equation}
where $X_i^{-1}X_j$ represents $x_j$ seen from $x_i$ and $Z_{ij}^{-1}$ represents $X_i$ as observed from $X_j$.

It turns out that $J$ is sparse, meaning that our linear system is sparse.

We recall that
\begin{equation}
\begin{aligned}
    b^T &= \sum\limits_{ij} b_{ij}^T &= \sum\limits_{ij} e_{ij}^T \Omega_{ij} J_{ij} \\
    H = \sum\limits_{ij} H_{ij} = \sum\limits_{ij} J_{ij}^T \Omega_{ij} J_{ij} \\
\end{aligned}
\end{equation}
$b$ will be dense, but $H$ will be dense, meaning we can solve the system efficiently.


\begin{algorithm}
	\SetKwInOut{Input}{input}\SetKwInOut{Output}{output}
	\Input{$x$}
	    \While{! converged }{
	         $(H,b) = buildLinearSystem(x)$ \;
	        $ \Delta x= solveSparse( H \Delta x = -b) $\;
	         $x = x + \Delta x$\;
	    }
	\Return{$x$}
	\caption{ $optimize(x)$}\label{alg:graphslam}
\end{algorithm}





\section{Solution Approaches}
\subsection{The Gauss-Newton Algorithm}

The Gauss-Newton algorithm is a heuristic algorithm for solving nonlinear least squares problems \cite{boyd:2018:vmls}. The algorithm is iterative, meaning it generates a series of points $x^{(1)}, x^{(2)}, \dots$. It involves alternating between two simple steps: (1) find an affine approximation of a function $f$, linearized at the current iterate $x^{(k)}$, and (2) solving the associated least squares problem to find the subsequent iterate $x^{(k+1)}$.

Formally, the affine approximation $\hat{f}$ is formed via Taylor approximation around $x^{(k)}$, as follows:
\begin{equation}
    \hat{f}(x; x^{(k)}) = f(x^{(k)}) + J(x^{(k)})(x-x^{(k)})
\end{equation}
where $J(x^{(k)}) = Df(x^{(k)})$ is the Jacobian of $f$ (the matrix of partial derivatives). The affine approximation is only a good locally, i.e. when $x$ is chosen close to $x^{(k)})$, meaning $\|x - x^{(k)}\|$ is small.

The subsequent iterate $x^{(k+1)}$ is chosen as
\begin{equation}
  x^{(k+1)} =  \mbox{arg } \underset{x}{\mbox{min}} \| \hat{f}(x; x^{(k)})\|^2
\end{equation}
If the Jacobian has linearly independent columns (i.e. is full rank), then we can directly compute the solution as
\begin{equation}
    x^{(k+1)} = x^{(k)} - \Bigg( J(x^{(k)})^T J(x^{(k)}) \Bigg)^{-1} J(x^{(k)})^T f(x^{(k)})
\end{equation}





\subsection{Levenberg-Marquardt Algorithm}

\subsection{Gauss-Newton for Pose Graphs}
Assuming the error has zero mean and is normally distributed, we Gaussian error with Information matrix $\Omega_i$
\begin{equation}
    x^{\star} = \mbox{arg} \underset{x}{\mbox{min}} \sum\limits_i e_i^T(x) \Omega_i e_i(x)
\end{equation}
where $x$ is the location of the robot and the location of the features (the map). The more certain we are about the error, the more we scale it via the Information matrix. This function is nonlinear, so we will use iterative numerical solutions. We start with assumptions: a good initial guess is available, error functions are smooth in the neighborhood of the global minima. Therfore, we assume we can solve the problem with iterative local linearizations.

Linearize the error erms around the current solution/initial guess
Compute the first derivative of the squared error function. Set it to zero and solve the linear system. We obtain a new state, hopefully closer to the minimum, and iterate.

\begin{equation}
e_i(x + \Delta x) \approx \vec{e}_i + J_i(x) \Delta x
\end{equation}
where 
\begin{equation}
    J_f(x) = \begin{bmatrix} \end{bmatrix}
\end{equation}

We formulate an expression for the global error:
\begin{equation}
\begin{aligned}
    \vec{e}_i(x) &= z_i - f_i(x) \\
    e_i(x) &= \vec{e}_i^T \Omega_i \vec{e}_i \\
    e_i(x) & \approx \big(\vec{e}_i + J_i(x) \Delta x\big)^T  \Omega_i \big(\vec{e}_i + J_i(x) \Delta x\big) \\
    e_i(x) & \approx \underbrace{\vec{e}_i^T \Omega_i \vec{e}_i}_{c_i} + \Delta x^T \underbrace{   J_i(x)^T  \Omega_i \vec{e}_i }_{b_i} +    \underbrace{ \vec{e}_i^T \Omega_i J_i(x)  }_{b_i^T}\Delta x +  \Delta x^T \underbrace{ J_i(x)^T  \Omega_i  J_i(x) }_{H_i} \Delta x \\
    e_i(x) & \approx c_i + 2b_i^T\Delta x + \Delta x^T H_i \Delta_x \\
    F(x) &= \sum\limits_i e_i(x) \\
    F(x) & \approx \sum\limits_i [c_i + 2b_i^T\Delta x + \Delta x^T H_i \Delta_x] \\
    F(x) &= \underbrace{\sum\limits_i c_i}_{c} + 2 \underbrace{\sum\limits_i b_i^T }_{b^T} \Delta x + \Delta x^T \underbrace{\sum\limits_i H_i}_{H} \Delta_x \\
    F(x) &= c + 2b^T\Delta x + \Delta x^T H \Delta x \\
    0 &= \frac{\partial F}{\partial x} = 2b + 2H \Delta x \\
    H \Delta x = -b \\
    \Delta x^{\star} = -H^{-1}b
\end{aligned}
\end{equation}
where $c_i$ is independent of $\Delta x$.
Which is a sum of scalars



## Review of Gauss-Newton Optimization

In vector calculus, the Jacobian matrix is the matrix of all first-order partial derivatives of a vector-valued function. In our case, the vector-valued function will be an error function. The entries of the Jacobian are defined as:

$$
J = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots 	& & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}
$$

To review Gauss-Newton optimization, which involves solving a series of least squares problems, please read my previous post [here](/gauss-newton/).




### Pose-Pose Constraints

Consider the pose of a robot at time $$i$$: $$\mathbf{x}_i = \begin{bmatrix} x_i \\ y_i \\ \theta_i \end{bmatrix}$$. If we wish to convert from this pose vector to a homogeneous transform, we can use the function $$v2t(\cdot)$$, defined as:

$$
v2t(\mathbf{x}_i) = \begin{bmatrix} R_i & \mathbf{t}_i \\ 0 & 1 \end{bmatrix} = \begin{bmatrix}
\mbox{cos}(\theta_i) & -\mbox{sin}(\theta_i) & x_i \\ 
\mbox{sin}(\theta_i) & \mbox{cos}(\theta_i) & y_i \\ 
0 & 0 & 1
\end{bmatrix} = {}^wX_i
$$

Consider the error function (boldface indicates a vector)

$$
\mathbf{e}_{ij} = t2v\bigg(Z_{ij}^{−1} ({}^wX_i^{−1} {}^wX_j )\bigg) = \begin{bmatrix}
R_{ij}^T\Big( R_i^T(\mathbf{t}_j - \mathbf{t}_i)-\mathbf{t}_{ij}\Big) \\
\theta_j - \theta_i - \theta_{ij}
\end{bmatrix} = \begin{bmatrix} \mathbf{e}_{xy} \\ \mathbf{e}_{\theta} \end{bmatrix}
$$, 

where $$Z_{ij} = v2t(\mathbf{z}_{ij})$$ is the
transformation matrix of the measurement $$\mathbf{z}_{ij}^T = (\mathbf{t}_{ij}^T, \theta_{ij})$$. We can use the measured $$\theta_{ij}$$ to form another rotation matrix $$R_{ij}$$. In our case the Jacobian of the error function $$\mathbf{e}_{ij}$$ is a $$3 \times 3$$ matrix, but will be simpler to write in 4 blocks, instead of 9 entries:

$$
A_{ij} = \frac{\partial \mathbf{e}_{ij}}{\partial \mathbf{x}_i} = \begin{bmatrix}
\frac{\partial \mathbf{e}_x}{\partial \mathbf{x}_{i_x}} & \frac{\partial \mathbf{e}_x}{\partial \mathbf{x}_{i_y}} & \frac{\partial \mathbf{e}_x}{\partial \mathbf{x}_{i_{\theta}}} \\
\frac{\partial \mathbf{e}_y}{\partial \mathbf{x}_{i_x}} & \frac{\partial \mathbf{e}_y}{\partial \mathbf{x}_{i_y}} & \frac{\partial \mathbf{e}_y}{\partial \mathbf{x}_{i_{\theta}}} \\
\frac{\partial \mathbf{e}_{\theta}}{\partial \mathbf{x}_{i_x}} & \frac{\partial \mathbf{e}_{\theta}}{\partial \mathbf{x}_{i_y}} & \frac{\partial \mathbf{e}_{\theta}}{\partial \mathbf{x}_{i_{\theta}}}
\end{bmatrix} = \begin{bmatrix}
\frac{\partial \mathbf{e}_{xy}}{\partial \mathbf{t}_i} &  \frac{\partial \mathbf{e}_{xy}}{\partial \mathbf{x}_{i_{\theta}}} \\
\frac{\partial \mathbf{e}_{\theta}}{\partial \mathbf{t}_i} & \frac{\partial \mathbf{e}_{\theta}}{\partial \mathbf{x}_{i_{\theta}}}
\end{bmatrix}
$$

We will derive each of these 4 Jacobian terms by hand using matrix calculus.

1. First, note that $$\mathbf{e}_{xy} = R_{ij}^T\Big( R_i^T(\mathbf{t}_j - \mathbf{t}_i)-\mathbf{t}_{ij}\Big)$$, so if we look at the terms that are multiplied with $$\mathbf{t}_i$$, we find
$$\frac{\partial \mathbf{e}_{xy}}{\partial \mathbf{t}_i} = R_{ij}^T R_i^T(-1)$$.

2. Second, consider
$$
\frac{\partial \mathbf{e}_{xy}}{\partial \mathbf{x}_{i_{\theta}}}
$$. Since we find $$R_i^T$$ in $$\mathbf{e}_{xy}$$, and $$R_i$$ is a function of $$\theta$$, i.e. $$R(\theta_i)$$, there is a dependence. Note that we can find the derivative of our 2D rotation matrix $$R_i^T = \begin{bmatrix} \mbox{cos}(\theta_i) & \mbox{sin}(\theta_i) \\ -\mbox{sin}(\theta_i) & \mbox{cos}(\theta_i) \end{bmatrix}$$. Formally, since $$R_i^T$$ is a function of $$\theta_i$$, the derivative $$\frac{\partial R_{i}^T}{\partial \theta_i}$$ is defined as:
$$
\frac{\partial R_{i}^T}{\partial \theta_i} = \begin{bmatrix} \frac{\partial \text{cos}(\theta_i)}{\partial \theta_i } & \frac{\partial \text{sin}(\theta_i)}{\partial \theta_i } \\ -\frac{\partial \text{sin}(\theta_i)}{\partial \theta_i } & \frac{\partial \text{cos}(\theta_i)}{\partial \theta_i } \end{bmatrix} = \begin{bmatrix} -\mbox{sin}(\theta_i) & \mbox{cos}(\theta_i) \\ -\mbox{cos}(\theta_i) & -\mbox{sin}(\theta_i) & \end{bmatrix}
$$

3. Third, consider
$$
\frac{\partial \mathbf{e}_{\theta}}{\partial \mathbf{t}_i} = 
$$
Since $$ \mathbf{e}_{\theta} = \theta_j - \theta_i - \theta_{ij}$$, there is no dependency between these $$\mathbf{e}_{\theta}$$ and $$\mathbf{t}_i$$, rendering this Jacobian entry zero.


4. Fourth, consider
$$
\frac{\partial \mathbf{e}_{\theta}}{\partial \mathbf{x}_{i_{\theta}}}.
$$
The dependency between $$\mathbf{e}_{\theta} = \theta_j - \theta_i - \theta_{ij}$$ and $$\mathbf{x}_{i_{\theta}}$$ is clear -- it is a scaling by $$-1$$.

We place these four partial derivatives into a matrix $$A_{ij}$$:


$$
A_{ij} = \frac{\partial \mathbf{e}_{ij}}{\partial \mathbf{x}_i} = \begin{bmatrix}
-R_{ij}^T R_i^T & R_{ij}^T \frac{\partial R_{i}^T}{\partial \theta_i} (\mathbf{t}_j - \mathbf{t}_i) \\
\mathbf{0}^T & -1
\end{bmatrix}
$$

$$
B_{ij} = \frac{\partial \mathbf{e}_{ij}}{\partial \mathbf{x}_j} = \begin{bmatrix}
R_{ij}^T R_i^T & \mathbf{0} \\
\mathbf{0}^T & 1
\end{bmatrix}
$$


### Pose-Landmark Constraints

Consider the position of a landmark $$\mathbf{x}_l = \begin{bmatrix} x_l \\ y_l \end{bmatrix}$$:

$$
\mathbf{e}_{il} = R_i^T  (\mathbf{x}_l − \mathbf{t}_i) − \mathbf{z}_{il}
$$

The Jacobian with respect to $$\mathbf{e}_{il}$$ is a $$(3 \times 2)$$ matrix that we can express with two blocks:

$$
\frac{ \partial \mathbf{e}_{il} }{\partial \mathbf{x}_i} = \begin{bmatrix} \frac{\partial \mathbf{e}}{\partial \mathbf{t}_i} & \frac{\partial \mathbf{e}}{\partial \mathbf{x}_{i_{\theta}}} \end{bmatrix}
$$

$$
A = \frac{ \partial \mathbf{e}_{il} }{\partial \mathbf{x}_i} = \begin{bmatrix} -R_i^T & \frac{\partial R_i^T}{\partial \theta} (\mathbf{x}_l - \mathbf{t}_i) \end{bmatrix}
$$


We will shorten $$\mathbf{e}_{il}$$ to $$\mathbf{e}$$ for ease of notation. Although we could write out all 4 terms of this Jacobian, i.e.

$$
\frac{\partial \mathbf{e}_{il}}{\partial \mathbf{x}_l} = \frac{\partial \mathbf{e}}{\partial \mathbf{x}_l}  = \begin{bmatrix}
\frac{\partial \mathbf{e}_x }{ \partial l_x} & \frac{\partial \mathbf{e}_x }{ \partial l_y} \\
\frac{\partial \mathbf{e}_x }{ \partial l_x} & \frac{\partial \mathbf{e}_x }{ \partial l_y}
\end{bmatrix}
$$

we can simply notice that $$\mathbf{x}_l$$ is multiplied only by $$R_i^T$$ in the error function. Thus,

$$
B = \frac{\partial \mathbf{e}_{il}}{\partial \mathbf{x}_l} = \frac{\partial \mathbf{e}}{\partial \mathbf{x}_l} = R_i^T
$$


## Additional Reading

You can find my Python code example in a gist [here](). Relevant Youtube lectures are here: [Video 1](https://www.youtube.com/watch?v=yVd-QDy0K6k&index=16&list=PLgnQpQtFTOGQrZ4O5QzbIHgl3b1JHimN_) and [Video 2](https://www.youtube.com/watch?v=VRGOLRGwAjg&list=PLgnQpQtFTOGQrZ4O5QzbIHgl3b1JHimN_&index=17).


## References

[1] Grisetti

[2] Dellaert

[3] Stephen Boyd and Lieven Vandenberghe





