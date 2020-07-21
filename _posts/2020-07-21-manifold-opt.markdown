---
layout: post
comments: true
permalink: /manifold-opt/
title:  "Optimization on Manifolds"
excerpt: "Pose3 SLAM, Bundle Adjustment"
date:   2020-07-21 11:00:00
mathjax: true
---


In nonlinear optimization, one important factor affecting the convergence is the mathematical structure of the object we are optimizing on. In many practical 3D robotics problems this is the $\mathrm{SE(3)}$ manifold describing the structure of 3D Poses.

It is not easy to directly operate on nonlinear manifolds like $\mathrm{SE(3)}$, so libraries like GTSAM uses the following strategy:
- Linearize the *error* manifold at the current estimate
- Calculate the next update in the associated tangent space
- Map the update back to the manifold with a *retract* map

We used two distinct but equally important concepts above: 1) the error metric, which is in a PoseSLAM problems is the measure of error between two poses; and 2) the *retract* operation, which is how we apply a computed linear update back to the nonlinear error manifold.

In GTSAM, you can choose, at compile time, between four different choices for the retract map on the $\mathrm{SE(3)}$ manifold:

- Full: Exponential map on $\mathrm{SE(3)}$
- Decomposed retract, which uses addition for translation and: 
  - Exponential map $\mathrm{SO(3)}$ with Rotation Matrix
  - Exponential map $\mathrm{SO(3)}$ with Quaternions
  - Cayley map on $\mathrm{SO(3)}$ with Rotation Matrix
  
  ## References
  [1] gtsam.org
  [2]
