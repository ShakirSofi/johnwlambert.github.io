---
layout: post
comments: true
permalink: /cam-calibration/
title:  "Camera Calibration"
excerpt: "Including Trust-Region Variant (Levenberg-Marquardt)"
date:   2018-03-31 11:00:00
mathjax: true
---


Table of Contents:
- [Least Squares](#back-substitution)
- [SVD](#lu)
- [Decomposing the Camera Matrix](#cholesky)


## Decomposing the Camera Matrix

Consider a $$3 \times 4$$ camera projection matrix $$M$$, with 11 degrees of freedom and all intrinsics and extrinsics combined:

$$
\begin{aligned}
M &= \begin{bmatrix} m_1 & m_2 & m_3 & m_4 \end{bmatrix} \\
M &= K [{}^cR_{w} | {}^{c}t_{w}] \\
M &= [K {}^cR_{w} | K {}^ct_{w}] \\
Q &= K {}^cR_{w} \\
Q^{-1} &= ({}^cR_{w})^{-1}K^{-1} \\
\end{aligned}
$$

Note that the object we wish is $${}^wt_c$$, which is the transformation from camera origin to world origin, meaning it is the camera center. Note that we can use the product $$K^{-1} K$$ to cancel the complicating $$K$$ factors:

$$
\begin{aligned}
{}^wt_c &= -R^Tt \\
{}^wt_c &= -Q^{-1} K {}^ct_{w} \\
{}^wt_c &= -R^{-1}K^{-1}Kt
\end{aligned}
$$

$$-R^Tt$$ 

This can also be obtained via an RQ decomposition of $$KR$$, since we wish to find the product of two matrices, where the rightmost matrix is orthogonal.



## Stereo Image Rectification


- Reprojectimage planes onto a common plane parallel to the line between camera centers

- Pixel motion is horizontal after this transforma7on

- Two homographies(3x3 transform), one for each input image reprojection

- C. Loop and Z. Zhang. Computing Rectifying Homographies for Stereo Vision. IEEE Conf. Computer Vision and PaWern Recognition (CVPR), 1999.


## Essential Matrix


$$
Eij = RTi (Ti − Tj )Rj 
$$

$$
Fij = K−Ti EijK−1j = Vi(Ti − Tj )VTj
$$
where $$Ti = [ti]×.$$

Kasten19iccv

## References

Yoni Kasten, Amnon Geifman, Meirav Galun, Ronen Basri. *Algebraic Characterization of Essential Matrices and Their Averaging in Multiview Settings*. ICCV 2019. [PDF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Kasten_Algebraic_Characterization_of_Essential_Matrices_and_Their_Averaging_in_Multiview_ICCV_2019_paper.pdf)




