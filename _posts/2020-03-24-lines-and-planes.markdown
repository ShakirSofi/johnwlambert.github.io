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
