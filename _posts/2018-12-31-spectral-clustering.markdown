---
layout: post
title:  "Spectral Clustering"
permalink: /spectral-clustering/
excerpt: " "
mathjax: true
date:   2018-12-27 11:00:00
mathjax: true

---
Table of Contents:
- [Need for Dimensionality Reduction](#need-for-dr)
- [PCA](#pca)


<a name='need-for-dr'></a>

## Spectral Clustering


\subsection{Graph Partitioning}
\begin{itemize}
\item Graph-cut problem: partition graph such that (1) edges between groups have a very low weight, and (2) edges within a group have high weight
\item Could use \textbf{Ratio Cut} (by size of the component) or the \textbf{Normalized Cut} (or by volume of the component, aka the sum of degrees)
\item Penalize using very small components, or very large components
\item These are NP-hard combinatorial problems. But Spectral clustering offers a way to solve the relaxation version of these problems
\item NCut -> use random-walk laplacian
\item RatioCut -> use unnormalized laplacian
\item 2-partition: assign by vector values of Fiedler vectors $\in \mathbbm{R}$, which ones $\in \mathbbm{R}_+$, or in $\mathbbm{R}_-$
\item Can apply k-means or standard clustering algorithm on the embedded points (transform graph clustering into a point clustering problem!). Take you into point cloud setting
\item K-means minimizes the distortion measure/energy
\begin{equation}
J = \sum\limits_j \sum\limits_k r_{ji}(y_j - \mu_i)^2 ??
\end{equation}
\item Centroid already minimizes the sum of squared distances
\item Can only reduce energy in every step (locally converge to minimum)
\item Can discover number of clusters that you need -- look at gap between eigenvalues, where is there a large gap $|\lambda_k - \lambda_{k-1}|$. 
\item Eigenvalues drop fast, then stabilize
\item Spectral clustering can cluster spirals of points, where k-nearest neighbors in this space would epicly fail
\item Look at data at a very different way... Even though data comes from a Euclidean space, easier to understand in the spectral space
\item edge weight $\mbox{exp} (-diff(pixel_{i}, pixel_j)/t^2 )$
\item Get reasonable, but not perfect, results for 3d segmentation
\item Fast and efficient with decent results
\end{itemize}

## References

[1] Leonidas Guibas. *Graph Laplacians, Laplacian Embeddings, and Spectral Clustering*. Lectures of CS233: Geometric and Topological Data Analysis, taught at Stanford University in Spring 2018.


