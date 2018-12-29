---
layout: post
title:  "Robot Localization"
permalink: /robot-localization/
excerpt: "ICP, grid histograms, ..."
mathjax: true
date:   2018-12-27 11:00:00
mathjax: true

---
Table of Contents:
- [A Basic SfM Pipeline](#sfmpipeline)
- [Cost Functions](#costfunctions)
- [Bundle Adjustment](#bundleadjustment)

<a name='sfmpipeline'></a>

## Localization

Navigation in urban environments requires clever solutions because sensors like GPS can't provide centimeter-level accuracy.  Precise localization requires fusing signals from sensors like GPS, IMU, wheel odometry, and LIDAR data in concert with pre-built maps [1].

In order for an autonomous robot to stay in a specific lane, it needs to know where the lane is. For an autonomous robot to stay in a lane, the localization requirements are in the order of decimeters [1].

I'll review some methods from 2007-2018.

## Building a Map

The dominant method is to learn a detailed map of the environment, and then to use a vehicle’s LIDAR sensor to localize relative to this map [1]. In [1], a "map" was a 2-D overhead view of the road surface, taken in the infrared spectrum with 5-cm resolution. This 2-D grid assigns to each x-y location in the environment an infrared reflectivity value. Thus, their ground map is a orthographic infrared photograph of the ground. To acquire such a map, multiple laser range finders are mounted on a vehicle, pointing downwards at the road surface. Obtain range + infrared reflectivity. GraphSLAM is employed to map roads.

To make a map, one has to differentiate between static and dynamic objects. There are two good ways to do this, as described in [1]
- track or remove objects that move at the time of mapping.
- reducing the map to features are very likely to be static. 

A 2D-world assumption is often not enough. In [2], localization is desired within a multi-level parking garage. They utilize multi-level surface maps to compactly represent such buildings, with a graph-based optimization procedure to establish the consistency of the map. Store surface height and variance $$\sigma_{ij}^{l}$$ to represent the uncertainty in the height of the surface. 

build an accurate map of the environment offline through aligning multiple
sensor passes over the same area. [9]


## Online Localization: Feature Matching in the Map

In the online stage, sensory input must be matched against the HD-map.

In [1], the authors correlate via the Pearson product-moment correlation the measured infrared reflectivity with the map. The importance weight of each particle is a function of the correlation

$$
w_t^{[k]} = \mbox{exp}\{ -\frac{1}{2} (x_t^{[k]} − y_t)^T \Gamma_t^{−1} (x_t^{[k]} − y_t)\} \cdot \Bigg( \mbox{corr} \Bigg[ \begin{pmatrix} h_1(m,x_t^{[k]}) \\ \vdots \\ h_{180}(m,x_t^{[k]}) \end{pmatrix}, \begin{pmatrix} z_t^1 \\ \vdots \\ z_t^{180} \end{pmatrix} \Bigg] + 1 \Bigg)
$$

In [2], online localization is performed via the iterative closest points (ICP) algorithm to obtain a maximum likelihood estimate of the robot motion between subsequent observations. Instead of performing ICP on raw 3D point clouds, the ICP is performed on local MLS-maps.

## Points Planes Poles Gaussian Bars over 2D Grids (Feature representation method)

 point-to-plane ICP between the raw 3D LiDAR points against the 3D pre-scanned localization prior at 10Hz
particle filter method for correlating LIDAR measurements [1]


## 2D grids

## Using Deep Learning

Barsan *et al.* are the first to use CNNs on LiDAR orthographic (bird's-eye-view) images of the ground for the online localization, feature matching step [9].  embeds both LiDAR intensity maps
and online LiDAR sweeps in a common space where calibration is not required. In this scenario, online localization can be performed by searching exhaustively over 3-DoF poses -- 2D position on the map manifold plus rotation. The authors score the quality of pose matches by the cross-correlation between the embeddings.

They use a histogram filter, the discretized Bayes' filter, to perform the search over 3D space. The three grid dimensions are $$x,y,\theta$$, and they compute the belief likelihood for every cell in the search space.

Barsan *et al.* generate ground-truth poses 
Our ground-truth poses are acquired through an expensive high
precision offline matching procedure with up to several centimeter uncertainty. We rasterize the
aggregated LiDAR points to create a LiDAR intensity image. Both the online intensity image and
the intensity map are discretized at a spatial resolution of 5cm covering a 30m×24m region.


## References

[1] J. Levinson, M. Montemerlo, and S. Thrun. *Map-based precision vehicle localization in urban environments*. In Robotics: Science and Systems, volume 4, page 1. Citeseer, 2007.

[2] R. Kummerle, D. Hahnel, D. Dolgov, S. Thrun, and W. Burgard. *Autonomous driving in a multi-level parking structure*. In IEEE International Conference on Robotics and Automation (ICRA), pages 3395–3400, May 2009.

[3] J. Levinson and S. Thrun. *Robust vehicle localization in urban environments using probabilistic maps*. In IEEE International Conference on Robotics and Automation (ICRA), 934 pages 4372–4378, May 2010.

[4] R. W. Wolcott and R. M. Eustice. *Fast LIDAR localization using multiresolution gaussian mixture maps*. In IEEE International Conference on Robotics and Automation (ICRA), pages 2814–2821, May 2015.

[5] R. W. Wolcott and R. M. Eustice. *Robust LIDAR localization using multiresolution gaussian mixture maps for autonomous driving*. The International Journal of Robotics Research, 36(3):292–319, 2017.

[6] H. Kim, B. Liu, C. Y. Goh, S. Lee, and H. Myung. *Robust vehicle localization using entropy-weighted particle filter-based data fusion of vertical and road intensity information for a large scale urban area*. IEEE Robotics and Automation 923 Letters, 2(3):1518–1524, July 2017. 

[7] R. Dub, M. G. Gollub, H. Sommer, I. Gilitschenski, R. Siegwart, C. Cadena, and J. Nieto. *Incremental-segment-based localization in 3-D point clouds*. IEEE Robotics and Automation Letters, 3(3):1832–1839, July 2018. 1

[8] G. Wan, X. Yang, R. Cai, H. Li, Y. Zhou, H. Wang, and S. Song. *Robust and precise vehicle localization based on multi-sensor fusion in diverse city scenes*. In IEEE International Conference on Robotics and Automation (ICRA), pages 4670–4677, May 2018.

[9] Ioan Andrei Barsan, Shenlong Wang, Andrei Pokrovsky, Raquel Urtasun. *Learning to Localize Using a LiDAR Intensity Map*. Proceedings of The 2nd Conference on Robot Learning, PMLR 87:605-616, 2018.

