

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

Suppose we are given a blocky "data" polyline (perhaps waypoints to traverse), but we desire a smooth trajectory through these waypoints. How can we find it? Linear interpolation will give us equally spaced points on the curve, but cubic interpolation will give us a smooth trajectory.

The Chord Length Method provides one solution, where the high-level goal is that *curve segments should not deviate much from the original inter-point distances*. Consider a taut string placed over the "data" polyline waypoints $$D_0, D_1, \dots, D_n$$. Between each pair of adjacent waypoints, a chord's length represents the length of that taut string. We compute each chord's length as the norm of each consecutive displacement vector $$\|D_i - D_{i-1}\|_2$$. Consider a polyline with $$n=4$$ waypoints:

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
From the Pythagorean theorem, we could also have deduced that the chord lengths would be $$\sqrt{4^2 + 1^2} = \sqrt{17} \approx 4.12$$, 1, and then $$\sqrt{2^2 + 1^2} = \sqrt{5} \approx 2.24$$. The length of the entire data polyline then is the sum of chord lengths $$L=\sum\limits_{i=1}^n \|D_i - D_{i-1}\|_2\|$$, or $$\approx 7.36$$. Our polyline domain is subdivided according to the distribution of the chord lengths. For a valid distribution, we normalize these "arclengths" to a unit total (we call them arc lengths since they will suggest the distance along each smooth curve/arc that we create):
```python
np.sum(chordlen)
7.36
chordlen = chordlen / np.sum(chordlen)
array([0.56, 0.14, 0.30 ])
```
The cumulative arc-lengths we achieve from the start to each respective point (as we traverse the points) should be $$L_k = \frac{ \sum\limits_{i=1}^k \|D_i - D_{i-1}\|_2 }{L}$$:
```python
cumarc = np.append(0, np.cumsum(chordlen))
array([0. , 0.56, 0.70 , 1. ])
```
A required input parameter to this algorithm is the number of points $$k$$ to sample the curve at. We'll generate $$k$$ equally spaced points, and suppose $$k=10$$. We want to know which interval each such point would fall within:
```python
k = 10
eq_spaced_points = np.linspace(0, 1, k)
array([0. , 0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 1. ])
>>> tbins = np.digitize(eq_spaced_points, cumarc)
array([1, 1, 1, 1, 1, 1, 2, 3, 3, 4])
```
As we might expect, most of our desired sampled points will fall onto the first (and longest) chord.

We'll need to clamp the 1-indexed intervals to [1,n-1] to prevent problems at the end where we might try to index into the 0th or n'th intervals, which don't exist:
```python
tbins[np.where((tbins <= 0) | (eq_spaced_points <= 0))] = 1
tbins[np.where((tbins >= n) | (eq_spaced_points >= 1))] = n - 1
```
We can now apply linear interpolation, simply traversing a portion of a straight line from the start waypoint:

```python
s = np.divide((eq_spaced_points - cumarc[tbins - 1]), chordlen[tbins - 1])
pt = pxy[tbins - 1, :] + np.multiply((pxy[tbins, :] - pxy[tbins - 1, :]), (np.vstack([s] * 2)).T)
```
<img width="569" src="https://user-images.githubusercontent.com/16724970/87857841-ad686a00-c8f7-11ea-9e6a-fe05c03a1d3e.png">


If instead we were to apply cubic interpolation, we would solve an ODE:



## Bezier Splines



## References

[1] Michael S. Floater and Tatiana Surazhsky. Parameterization for curve interpolation. 2005. [PDF](https://www.mn.uio.no/math/english/people/aca/michaelf/papers/curve_survey.pdf)

[1] Ching-Kuang Shene. [Link](https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/INT-APP/PARA-chord-length.html).

[2] John D'Errico. https://www.mathworks.com/matlabcentral/fileexchange/34874-interparc
