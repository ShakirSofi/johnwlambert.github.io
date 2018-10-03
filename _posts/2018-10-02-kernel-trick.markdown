---
layout: post
title:  "Kernel Trick"
permalink: /kernel-trick/
excerpt: "The Kernel Trick is a poorly taught but beautiful piece of insight that makes SVMs work."
mathjax: true
date:   2018-10-02 11:01:00
mathjax: true

---

## The Kernel Trick
The Kernel Trick is a poorly taught but beautiful piece of insight that makes SVMs work.

$$
K(x,z) = (x^Tz)^2 = (\sum\limits_{i=1}^n x_iz_i)^2
$$

You might be tempted to simplify this immediately to \\( \sum\limits_{i=1}^n (x_iz_i)^2 \\), but this would be incorrect. This expression is actually a square of sums, which involves a large product (remember expanding binomial products like \\( (a-b)(a-b) \\) which involve \\(2^2\\) terms ).

We can expand the square of sums, and we'll use different indices \\(i,j\\) to keep track of the respective terms:

$$
K(x,z) = (\sum\limits_{i=1}^n x_iz_i) (\sum\limits_{j=1}^n x_jz_j)
$$

$$
K(x,z) = \sum\limits_{i=1}^n \sum\limits_{j=1}^n (x_ix_j) (z_iz_j)
$$


This is much easier to understand with an example. Consider \\(n=3\\):

$$
K(x,z) = \begin{bmatrix} x_1z_1 + x_2z_2 + x_3z_3 \end{bmatrix} \begin{bmatrix} x_1z_1 + x_2z_2 + x_3z_3 \end{bmatrix}
$$

Expanding terms, we get a sum with 9 total terms:

$$ 
K(x,z) = \bigg(x_1z_1x_1z_1 +  x_1z_1x_2z_2 + x_1z_1x_3z_3\bigg) + \bigg(x_2z_2 x_1z_1 +  x_2z_2 x_2z_2 +   x_2z_2x_3z_3\bigg) + \bigg( x_3z_3x_1z_1 + x_3z_3x_2z_2 + x_3z_3x_3z_3 \bigg)
$$

Surprisingly, this sum can be written as a simple inner product of two vectors

$$
K(x,z) =\phi(x)^T \phi(z) = \begin{bmatrix} x_1x_1  \\ x_1x_2 \\ x_1x_3 \\ x_2x_1 \\ x_2x_2 \\ x_2x_3  \\ \vdots  \\ \end{bmatrix} \begin{bmatrix} z_1z_1  \\ z_1z_2 \\ z_1z_3 \\ z_2z_1 \\ z_2z_2 \\ z_2z_3  \\ \vdots  \\ \end{bmatrix}
$$

Thus, we've shown that for \\(n=3\\),
$$
K(x,z) = \phi(x)^T \phi(z)
$$
which contains all of the cross-product terms.

