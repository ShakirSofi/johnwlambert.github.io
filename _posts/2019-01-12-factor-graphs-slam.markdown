---
layout: post
title:  "Factor Graphs for SLAM and SfM"
permalink: /factor-graphs/
excerpt: "Gaussian densities, direct methods, iterative methods"
mathjax: true
date:   2018-12-28 11:00:00
mathjax: true

---
Table of Contents:
- [Factor Graphs](#whyliegroups)
- [Factor Densities](#liegroups)
- [MAP Inference on Factor Graphs](#son)
- [Factor Graph Variable Elimanation](#so2)

<a name='whyliegroups'></a>

## Factor Graphs for Robot Vision

A factor graph is a probabilistic graphical model, theory which represents the marriage of graph theory with probability.

To use factor graphs for computer vision in robotics, we will need another tool: numerical linear algebra. These problems can be reformulated as very large least-squares problems. In fact, the size of the matrix that would not need be to inverted to solve the system of equations makes these problems solvable only by iterative methods, rather than direct methods.

## Factor Graphs: Background

A factor graph is a probabilistic graphical model (in the same family with Markov Random Fields (MRFs) and Bayesian Networks). It is an undirected graph (meaning there are no parents or topological ordering).

Bayesian Networks are directed graphs where edges in the graph are associated with conditional probability distributions (CPDs), assigning the probability of children in the graph taking on certain values based on the values of the parents.

In undirected models like MRFs and Factor Graphs, instead of specifying CPDs, we specify (non-negative) potential functions (or factors) over sets of variables associated with cliques (complete subgraphs) C of the graph.  Like Conditional Prob. Distributions, a factor/potential can be represented as a table, but it is not normalized (does not sum to one). 

A factor graph is a bipartite undirected graph with variable nodes (circles) and factor nodes (squares). Edges are only between the variable nodes and the factor nodes.

The variable nodes can take on certain values, and the likelihood of that event for a set of variables is expressed in the potential (factor node) attached to those variables.  Each factor node is associated with a single potential, whose scope is the set of variables that are neighbors in the factor graph.

A small example might make this clearer. Suppose we have a group of four people: Alex, Bob, Catherine, David A=Alex’s hair color (red, green, blue)

B=Bob’s hair color

C=Catherine’s hair color

D=David’s hair color

Alex and Bob are friends, Bob and Catherine are friends, Catherine and David are friends, David and Alex are friends

Friends never have the same hair color!

 

It turns out that this distribution p cannot be represented (perfectly) by any Bayesian network. But it is succinctly represented by a Factor Graph or MRF.
https://d1b10bmlvqabco.cloudfront.net/attach/jl1qtqdkuye2rp/jl1r1s4npvog2/jmsma7unw07s/Screen_Shot_20181002_at_11.48.05_PM.png

The Factor Graph distribution is same as the MRF – this is just a different graph data structure.

## Factor Densities

The most often used probability densities involve the multi-variate Gaussian distribution [1], whose density is given by


$$
x \sim p(x) = \mathcal{N}(\mu, \Sigma) = 
\frac{1}{\sqrt{(2\pi)^n|\Sigma|}}\mbox{exp}
\Big\{ -\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \Big\}
$$

For the sake of brevity, we can write this as:

$$
x \sim p(x) = \mathcal{N}(\mu, \Sigma) = 
\frac{1}{\sqrt{(2\pi)^n|\Sigma|}}\mbox{exp}
\Big\{ -\frac{1}{2} \|(x - \mu)^T \|_{\Sigma^{-1}}^2 \Big\}
$$


## MAP Inference on Factor Graphs

MAP inference in SLAM is exactly the process of determining those
values for the unknowns that maximally agree with the information
present in the uncertain measurements [1]


We wish to solve for $$X$$, the position of the robot at all timesteps.

$$
\begin{aligned}
X^{MAP} &= \mbox{arg }\underset{X}{\mbox{max }} \phi(X) & \mbox{Maximum likelihood Defn.} \\
 &= \mbox{arg }\underset{X}{\mbox{max }} \prod\limits_i \phi_i(X_i) & \mbox{text} \\
  &= \mbox{arg }\underset{X}{\mbox{max }} \prod\limits_i \mbox{exp } \bigg\{ -\frac{1}{2}\|h_i(X_i) - z_i\|_{\Sigma_i}^2 \bigg\} & \mbox{Use Gaussian prior and likelihood factors} \\
    &= \mbox{arg }\underset{X}{\mbox{max }} \mbox{log } \prod\limits_i  \mbox{exp } \bigg\{ -\frac{1}{2}\|h_i(X_i) - z_i\|_{\Sigma_i}^2 \bigg\} & \mbox{Log is monotonically increasing} \\
     &= \mbox{arg }\underset{X}{\mbox{max }} \sum\limits_i \mbox{log }   \mbox{exp } \bigg\{ -\frac{1}{2}\|h_i(X_i) - z_i\|_{\Sigma_i}^2 \bigg\} & \mbox{Log of a product is a sum of logs.} \\
       &= \mbox{arg }\underset{X}{\mbox{max }} \sum\limits_i  \bigg( -\frac{1}{2}\|h_i(X_i) - z_i\|_{\Sigma_i}^2 \bigg) & \mbox{Simplify } f(f^{-1}(x))=x \\
        &= \mbox{arg }\underset{X}{\mbox{min }} \sum\limits_i  \bigg( \frac{1}{2}\|h_i(X_i) - z_i\|_{\Sigma_i}^2 \bigg) & \mbox{Max of } -f(x) \mbox{ is min of } f(x) \\
\end{aligned}
$$

We've reduce the problem to minimization of a sum of nonlinear least-squares.

Along the way, we decided to maximize the log-likelihood instead of the likelihood because the argument that maximizes the log of a function also maximizes the function itself.


## Factor Graph Variable Elimanation

If you're wondering about the variable elimination part, we choose subsets of variables connected by factors and start combining them by taking the product of their factors and marginalizing out variables.

 
In the table above, we have variables as the columns and factors as the rows. We can combines factors progressively to involve more and more variables.

https://d1b10bmlvqabco.cloudfront.net/attach/jl1qtqdkuye2rp/jl1r1s4npvog2/jmtgs7d71g2p/Screen_Shot_20181003_at_2.05.56_PM.png




## References

[1] Frank Dellaert and Michael Kaess. *Factor Graphs for Robot Perception*. Foundations and Trends in Robotics, Vol. 6, No. 1-2 (2017) 1–139. [PDF](https://www.ri.cmu.edu/wp-content/uploads/2018/05/Dellaert17fnt.pdf).

[2] Stefano Ermon. 


http://videolectures.net/mlss06tw_roweis_mlpgm/?q=sam%20roweis


https://www.cc.gatech.edu/~bboots3/STR-Spring2018/readings/graphical_model_Jordan.pdf


