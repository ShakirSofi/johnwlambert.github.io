---
layout: post
title:  "Kernel Trick"
permalink: /kernel-trick/
mathjax: true
date:   2018-10-02 11:01:00
mathjax: true

---
Table of Contents:
- [Kernel Trick](#rank)


## The Kernel Trick
The Kernel Trick is a poorly taught but beautiful piece of insight that makes SVMs work.

$$
K(x,z) = (x^Tz)^2 = (\sum\limits_{i=1}^n x_iz_i)^2
$$

You might be tempted to simplify this immediately to, but this square of sums, which involves a large product (Remember expanding binomial products like \\( (a-b)(a-b) \\) ).

We can expand the square of sums, and we'll use different indices \\(i,j\\) to keep track of the respective terms:

$$
=(\sum\limits_{i=1}^n x_iz_i)^2 (\sum\limits_{j=1}^n x_jz_j)^2
$$

$$
=\sum\limits_{i=1}^n \sum\limits_{j=1}^n (x_ix_j) (z_iz_j)
$$


This is much easier to understand with an example. Consider \\(n=3\\):

$$
\begin{bmatrix} x_1z_1 + x_2z_2 + x_3z_3 \end{bmatrix} \begin{bmatrix} x_1z_1 + x_2z_2 + x_3z_3 \end{bmatrix}
$$

Expanding terms, we get a sum with 9 total terms:

$$ 
= (x_1z_1x_1z_1 +  x_1z_1x_2z_2 + x_1z_1x_3z_3) + (x_2z_2 x_1z_1 +  x_2z_2 x_2z_2 +   x_2z_2x_3z_3) +  x_3z_3x_1z_1 + x_3z_3x_2z_2 + x_3z_3x_3z_3
$$

Surprisingly, this sum can be written as a simple inner product of two vectors

$$
=\phi(x)^T \phi(z) = \begin{bmatrix} x_1x_1  \\ x_1x_2 \\ x_1x_3 \\ x_2x_1 \\ x_2x_2 \\ x_2x_3  \\ \vdots  \\ \end{bmatrix} \begin{bmatrix} z_1z_1  \\ z_1z_2 \\ z_1z_3 \\ z_2z_1 \\ z_2z_2 \\ z_2z_3  \\ \vdots  \\ \end{bmatrix}
$$

Thus, we've shown that for \\(n=3\\),
$$
K(x,z) = \phi(x)^T \phi(z)
$$

which contains all of the cross-product terms.

