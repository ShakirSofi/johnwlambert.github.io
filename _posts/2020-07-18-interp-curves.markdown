

---
layout: post
comments: true
permalink: /interp-curves/
title:  "Interpolation and Curves"
excerpt: "linear interpolation, chord length interpolation, Bezier splines"
date:   2020-07-18 11:00:00
mathjax: true
---


## Interpolation



## Chord Length Method

Suppose we are given a blocky "data" polyline (perhaps waypoints to traverse), but we desire a smooth trajectory through these waypoints. How can we find it?

The Chord Length Method provides one solution, where the high-level goal is that *curve segments should not deviate much from the original inter-point distances*. Consider a taut string placed over the "data" polyline waypoints $$D_0, D_1, \dots, D_n$$. Between each pair of adjacent waypoints, a chord's length represents the length of that taut string. We compute each chord's length as the norm of each consecutive displacement vector $$\|D_i - D_{i-1}\|_2$$

```python
>>> pxy = np.array(
	[
		[0, 0],
		[1, 4],
		[2, 4],
		[3, 2]
	])
diff = np.diff(pxy,axis=0)
array([[ 1,  4],
       [ 1,  0],
       [ 1, -2]])
chordlen = np.linalg.norm(diff, axis=1)
array([4.12, 1.        , 2.24])
```
From the Pythagorean theorem, we could also have deduced that the chord lengths would be $$\sqrt{4^2 + 1^2} = \sqrt{17} \approx 4.12$$, 1, and then $$\sqrt{2^2 + 1^2} = \sqrt{5} \approx 2.24$$. The length of the entire data polyline then is the sum of chord lengths $$L=\sum\limits_{i=1}^n \|D_i - D_{i-1}\|_2\|$$, or $$\approx 7.36$$. Our polyline domain is subdivided according to the distribution of the chord lengths. For a valid distribution, we normalize these "arclengths" to a unit total
```python
np.sum(chordlen)
7.36
chordlen = chordlen / np.sum(chordlen)
array([0.56, 0.14, 0.30 ])
```
The cumulative arc-lengths we achieve as we traverse the points should be:
```python
cumarc = np.append(0, np.cumsum(chordlen))
array([0. , 0.56, 0.70 , 1. ])
```

## Bezier Splines



## References

[1] Ching-Kuang Shene. https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/INT-APP/PARA-chord-length.html
