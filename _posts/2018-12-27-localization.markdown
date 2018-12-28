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

I'll review some methods from 2007-2018.

## Points Planes Poles Gaussian Bars over 2D Grids (Feature representation method)


particle filter method for correlating LIDAR measurements [1]


## 2D grids

## Using Deep Learning

[1] J. Levinson, M. Montemerlo, and S. Thrun. Map-based 929 precision vehicle localization in urban environments. In 930
Robotics: Science and Systems, volume 4, page 1. Citeseer,
2007. 1, 2, 5 931

[2] R. Kummerle, D. Hahnel, D. Dolgov, S. Thrun, and W. Bur- 925 gard. Autonomous driving in a multi-level parking structure. 926 In IEEE International Conference on Robotics and Automa- 927 tion (ICRA), pages 3395–3400, May 2009. 1

[3] J. Levinson and S. Thrun. Robust vehicle localization in 932 urban environments using probabilistic maps. In IEEE In- 933 ternational Conference on Robotics and Automation (ICRA), 934 pages 4372–4378, May 2010. 1, 2, 5, 7

[4] R. W. Wolcott and R. M. Eustice. Fast LIDAR localization using multiresolution gaussian mixture maps. In IEEE In- ternational Conference on Robotics and Automation (ICRA), pages 2814–2821, May 2015. 1, 2

[5] R. W. Wolcott and R. M. Eustice. Robust LIDAR local- ization using multiresolution gaussian mixture maps for au- tonomous driving. The International Journal of Robotics Re- search, 36(3):292–319, 2017. 1, 2

[6] H. Kim, B. Liu, C. Y. Goh, S. Lee, and H. Myung. Robust 920 vehicle localization using entropy-weighted particle filter- 921 based data fusion of vertical and road intensity information 922 for a large scale urban area. IEEE Robotics and Automation 923 Letters, 2(3):1518–1524, July 2017. 

[7] R. Dub, M. G. Gollub, H. Sommer, I. Gilitschenski, R. Sieg- wart, C. Cadena, and J. Nieto. Incremental-segment-based localization in 3-D point clouds. IEEE Robotics and Au- tomation Letters, 3(3):1832–1839, July 2018. 1

[8] G. Wan, X. Yang, R. Cai, H. Li, Y. Zhou, H. Wang, and S. Song. Robust and precise vehicle localization based on multi-sensor fusion in diverse city scenes. In IEEE Inter- national Conference on Robotics and Automation (ICRA), pages 4670–4677, May 2018.

