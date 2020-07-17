---
layout: default
---

|![](/images/mom_melissa_huntington_small.jpg) | Ph.D. Candidate<br>[School of Interactive Computing](https://www.ic.gatech.edu/)<br>[Georgia Institute of Technology](https://www.gatech.edu/) | 

## Bio

I am a Ph.D. student at Georgia Tech, where I have the good fortune to work with Professors [James Hays](https://www.cc.gatech.edu/~hays/) and [Frank Dellaert](http://frank.dellaert.com/). I completed my Bachelor's and Master's degrees in Computer Science at Stanford University in 2018, specializing in artificial intelligence.  

You can reach me at johnlambert AT gatech DOT edu. Some of my code can be [found here](http://github.com/johnwlambert/).


[[My CV]](/assets/cv.pdf)


## News
- June 2020: Watch an [excellent presentation from MachinesCanSee](https://youtu.be/lwv85fC1Ids) on our recent [MSeg](http://vladlen.info/papers/MSeg.pdf) work, presented by Dr. Vladlen Koltun.
- June 2020: The CVPR 2020 WAD Argoverse competitions have concluded. Congratulations to the very impressive submissions from ths winners. You can watch the results presentation [here](https://www.youtube.com/watch?v=Vcbj_peZT4Q&feature=youtu.be), or the summary presented at [ICML 20](https://slideslive.com/38930752/what-we-learned-from-argoverse-competitions).
- April 2020: We are please to announce two [Argoverse](https://www.argoverse.org/tasks.html) Competitions at the [CVPR 2020 Workshop on Autonomous Driving](http://cvpr2020.wad.vision/). Argo AI is offering $5,000 in prizes for Motion Forecasting and 3D tracking methods. I've open-sourced [my 3d tracking code](https://github.com/johnwlambert/argoverse_cbgs_kf_tracker) that is currently 1st place on the [leaderboard](https://evalai.cloudcv.org/web/challenges/challenge-page/453/overview). Please consider participating! The competitions will remain open until June 10, 2020.
- April 2020: Our [MSeg](http://vladlen.info/papers/MSeg.pdf) paper has been accepted to CVPR 2020 and took first place on [WildDash](https://wilddash.cc/benchmark/summary_tbl?hc=semantic_rob). Pretrained models available [here](https://github.com/mseg-dataset/mseg-semantic), data available [here](https://github.com/mseg-dataset/mseg-api), and a [Colab](https://colab.research.google.com/drive/1ctyBEf74uA-7R8sidi026OvNb4WlKkG1?usp=sharing) to try our demo on your own images and videos.

## Research
Humans have an amazing ability to understand the world through their visual system but designing automated systems to perform the task continues to prove difficult. We take for granted almost everything our visual system is capable of. While great progress has been made in 2D image understanding, the real world is 3D, not 2D, so reasoning in the 2D image plane is insufficient. The 3D world is high-dimensional and challenging and has a high data requirement.

My research interests revolve around geometric and semantic understanding of 3D environments. Accurate understanding of 3D environments will have enormous benefit for people all over the world, with implications for safer transportation and safer workplaces.

## Teaching

Aside from research, another passion of mine is teaching.
I enjoy creating teaching materials for topics related to computer vision, a field which relies heavily upon numerical optimization and statistical machine learning tools. A number of teaching modules I've written can be found below:

<div class="teaching-home">
  <div class="materials-wrap">

    <div class="module-header">Module 1: Linear Algebra</div>

    <div class="materials-item">
      <a href="linear-algebra/">
        Linear Algebra Without the Agonizing Pain
      </a>
      <div class="kw">
        Necessary Linear Algebra Overview
      </div>
    </div>

    <div class="materials-item">
      <a href="fast-nearest-neighbor/">
         Fast Nearest Neighbors
      </a>
      <div class="kw">
        Vectorizing nearest neighbors (with no for-loops!)
      </div>
    </div>

    <div class="module-header">Module 2: Numerical Linear Algebra </div>
    
    <div class="materials-item">
      <a href="direct-methods/">
         Direct Methods for Solving Systems of Linear Equations
      </a>
      <div class="kw">
        backsubstitution and the LU, Cholesky, QR factorizations
      </div>
    </div>

    <div class="materials-item">
      <a href="cg-orthomin/">
         Conjugate Gradients
      </a>
      <div class="kw">
        large systems of equations, Krylov subspaces, Cayley-Hamilton Theorem
      </div>
    </div>

    <div class="materials-item">
      <a href="least-squares/">
         Least-Squares
      </a>
      <div class="kw">
        QR decomposition for least-squares, modified Gram-Schmidt, GMRES
      </div>
    </div>


    <div class="module-header">Module 3: SVMs and Optimization </div>

    <div class="materials-item">
      <a href="kernel-trick/">
        The Kernel Trick
      </a>
      <div class="kw">
          poorly taught but beautiful piece of insight that makes SVMs work
      </div>
    </div>

    <div class="materials-item">
      <a href="gauss-newton/">
        Gauss-Newton Optimization in 10 Minutes
      </a>
      <div class="kw">
        Including Trust-Region Variant (Levenberg-Marquardt)
      </div>
    </div>


    <div class="materials-item">
      <a href="convex-opt/">
        Convex Optimization Without the Agonizing Pain
      </a>
      <div class="kw">
        Constrained Optimization, Lagrangians, Duality, and Interior Point Methods
      </div>
    </div>

    <div class="materials-item">
      <a href="subgradient-methods/">
        Subgradient Methods in 10 Minutes
      </a>
      <div class="kw">
        Convex Optimization Part II
      </div>
    </div>

    <div class="module-header">Module 4: State Estimation </div>

    <div class="materials-item">
      <a href="bayes-filter/">
        What is State Estimation? and the Bayes Filter
      </a>
      <div class="kw">
          linear dynamical systems, bayes rule, bayesian estimation, and filtering
      </div>
    </div>

    <div class="materials-item">
      <a href="lie-groups/">
        Lie Groups and Rigid Body Kinematics
      </a>
      <div class="kw">
          SO(2), SO(3), SE(2), SE(3), Lie algebras
      </div>
    </div>


    <div class="module-header">Module 5: Geometry and Camera Calibration </div>


    <div class="materials-item">
      <a href="stereo/">
        Stereo and Disparity
      </a>
      <div class="kw">
          disparity maps, cost volume, MC-CNN
      </div>
    </div>


    <div class="materials-item">
      <a href="epipolar-geometry/">
        Epipolar Geometry and the Fundamental Matrix
      </a>
      <div class="kw">
          simple ideas that are normally poorly explained 
      </div>
    </div>

    <div class="materials-item">
      <a href="icp/">
        Iterative Closest Point
      </a>
      <div class="kw">
          simple derivations and code examples 
      </div>
    </div>

    <div class="module-header">Module 6: Reinforcement Learning </div>

    <div class="materials-item">
      <a href="policy-gradients/">
        Policy Gradients
      </a>
      <div class="kw">
          intuition and simple derivations of REINFORCE, TRPO
      </div>
    </div>


    <div class="module-header">Module 7: Convolutional Neural Networks </div>

    <div class="materials-item">
      <a href="pytorch-tutorial/">
        PyTorch Tutorial
      </a>
      <div class="kw">
          PyTorch tensor operations, initializing CONV layers, groups, custom modules
      </div>
    </div>

    <div class="module-header">Module 8: Geometric Data Analysis </div>

    <div class="module-header">Module 9: Message Passing Interface (MPI) </div>
 
  </div>
</div>



