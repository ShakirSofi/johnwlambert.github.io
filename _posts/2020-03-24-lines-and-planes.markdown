---
layout: post
title:  "Lines and Planes"
permalink: /lines-planes/
excerpt: ", ..."
mathjax: true
date:   2020-03-24 11:00:00
mathjax: true

---
Table of Contents:
- [Lines](#sfmpipeline)
- [Planes](#costfunctions)


Basic Geometry: Lines and Planes


## Lines


https://www.topcoder.com/community/competitive-programming/tutorials/geometry-concepts-line-intersection-and-its-applications/


https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect

Cross products and determinant


## Point to Line Distance: Via Cross Product
We'll use intuition about parallelograms to prove this.

Area of Parallelogram $$= \| \vec{AB} \times \vec{AP} \|_2 = \mbox{base } \cdot \mbox{ height}$$, since every parallelogram can be made into a rectangle (since base and height are perpendicular).
height of parallelogram is the distance from point to line
base is \|AB\|_2

Thus, $$= \| \vec{AB} \times \vec{AP} \|_2 = \| \vec{AB} \|_2 \cdot height $$
$$ height = \frac{ \| \vec{AB} \times \vec{AP} \|_2  }{ \| \vec{AB} \|_2 } $$
Thus, dist(p, AB)

```python
def point_to_line_dist(a,b,p):
  ab = b - a
  ap = p - a
  return np.linalg.norm(np.cross(ap,ab)) / np.linalg.norm(ab)
```

## Planes

Parameterization

SVD

Generating points on a plane


# Ray-Plane Intersection 

$$
\begin{aligned}
P = O + tR\\
Ax + By + Cz + D = 0\\
A * P_x + B * P_y + C * P_z + D = 0\\
A * (O_x + tR_x) + B * (O_y + tR_y) + C * (O_z + tR_z) + D = 0\\
A * O_x + B * O_y + C * O_z + A * tR_x + B * tR_y + C * tR_z + D = 0\\
t * (A * R_x + B * R_y + C * R_z) + A * O_x + B * O_y + C * O_z + D = 0\\
t = -{\dfrac{A * O_x + B * O_y + C * O_z + D}{A * R_x + B * R_y + C * R_z}}\\
t = -{\dfrac{ N(A,B,C) \cdot O + D}{N(A,B,C) \cdot R}}
\end{aligned}
$$
Ref: https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution

# Moller-Trombore

Cramer's Rule

https://math.stackexchange.com/questions/1941590/how-does-cramers-rule-work

I explain it here in two variables, but the principle is the same.

Say you have an equation

$$
\begin{pmatrix}a&b\\c&d\end{pmatrix}\begin{pmatrix}x\\y\end{pmatrix}=\begin{pmatrix}p\\q \end{pmatrix}
$$

Now you can see that the following holds

$$
\begin{pmatrix}a&b\\c&d\end{pmatrix}\begin{pmatrix}x&0\\y&1\end{pmatrix}=\begin{pmatrix}p&b\\q &d\end{pmatrix}
$$
Finally just take the determinant of this last equation; det is multiplicative so you get
\Delta x=\Delta_1

where |A| = \Delta
and |A_1| = \Delta_1

and x=\frac{\Delta_1}\Delta

