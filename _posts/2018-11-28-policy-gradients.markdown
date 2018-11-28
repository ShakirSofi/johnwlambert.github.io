---
layout: post
comments: true
permalink: /policy-gradients/
title:  "Understanding Policy Gradients"
excerpt: "Including Trust-Region Variant (Levenberg-Marquardt)"
date:   2018-03-31 11:00:00
mathjax: true

---






## Policy Gradients

As opposed to Deep Q-Learning, policy gradients is a method to directly output probabilities of actions. We scale gradients of actions by reward.

Pros:
-  A **policy is often easier to approximate than value function**. For example, if playing Pong, it is hard to assign a specific score to moving your paddle up vs. down.  
- Policy Gradients **works well empirically** and was a key to AlphaGo's success.
- Policy gradients **learns stochastic optimal policies**, which is crucial for many applications. For example, in the game of *Rock, Paper, Scissors*, a deterministic policy is easily exploited, but a uniform random policy is optimal.

Cons:
- Training takes forever. The use of sampled data is not efficient. High variance confounds actions. Need tons of data for estimator to be good enough.
- Converge to local optima. Often there are many optima.
- Unlike human learning: humans can use rapid, abstract model building.


### Background 


### Geometric Intuition

Suppose we have a function \\(f(x)\\), such that \\(f(x) \geq 0 \forall x\\)
For every \\(x_i\\), the gradient estimator \\( \hat{g}_i\\) tries to push up on its density.


<div class="fig figcenter fighighlight">
  <img src="/assets/policy_gradients_geometric_intuition.png" width="65%">
  <div class="figcaption">
    With higher function values f(x) to the right, the probability density p(x) will be pushed up by vectors with higher magnitude on the right.  Image source: John Schulman [2].
  </div>
</div>

### Math Background
To understand the proofs, you'll need to understand 3 simple tricks:

- The derivative of the natural logarithm: \\( \nabla_{\theta} \mbox{log }z = \frac{1}{z} \nabla_{\theta} z \\)
- The definition of expectation:

$$
\begin{align}
\mathbb{E}_{x \sim p(x)}[f(x)] = \sum\limits_x p(x) f(x) \\
\mathbb{E}_{x \sim p(x)}[f(x)] = \int_x p(x) f(x) \,dx
\end{align}
$$

- Multiplying a fraction’s numerator and denominator by some arbitrary, non-zero constant does not change the fraction’s value.

$$
\frac{a}{b} = \frac{a \cdot p(x)}{b \cdot p(x)}
$$

## Policy Gradient Theorem


Let \\(x\\) be the action we will choose, \\(s\\) be the parameterization of our current state, \\(\theta\\) be the weights of our neural network. Then our neural network will directly output probabilities \\(p(x \mid s; \theta)\\) that depend upon the input (\\(s\\)) and the network weights (\\( \theta \\)).

We start with a desire to maximize our expected reward. Our reward function is \\(r(x)\\), which is dependent upon the action \\(x\\) that we take.

$$
\begin{align}
\nabla_{\theta} \mathbb{E}_{x \sim p( x \mid s; \theta)} [r(x)] &= \nabla_{\theta} \sum_x p(x \mid s; \theta) r(x) & \text{Defn. of expectation} \\
& = \sum_x r(x) \nabla_{\theta} p(x \mid s; \theta) & \text{Push gradient inside sum} \\
& = \sum_x r(x) p(x \mid s; \theta) \frac{\nabla_{\theta} p(x \mid s; \theta)}{p(x \mid s; \theta)}  & \text{Multiply and divide by } p(x \mid s; \theta) \\
& = \sum_x r(x) p(x \mid s; \theta) \nabla_{\theta} \log p(x \mid s; \theta) & \text{Apply } \nabla_{\theta} \log(z) = \frac{1}{z} \nabla_{\theta} z \\
& = \mathbb{E}_{x \sim p( x \mid s; \theta)} [r(x) \nabla_{\theta} \log p(x \mid s; \theta) ] & \text{Definition of expectation}
\end{align}
$$

A similar derivation can be found in [1] or in [2].

What does this mean for us? It means that if you want to maximize your expected reward, you could do something like gradient ascent. It turns out that the gradient of the expected reward is simple to compute and analytical -- it is simply **the expectation of the reward times the log probabilities of actions**.

## Consider Trajectories
Now, let's consider trajectories, which are sequences of actions. 

The *Markov Property* states that Markov Property: the next state \\(s^{\prime}\\) depends on the current state \\(s\\) and the decision maker’s action \\(a\\). But given \\(s\\) and \\(a\\), it is conditionally independent of all previous states and actions.

The probability of a trajectory is defined as:

$$
p(\tau \mid \theta) = \underbrace{ \mu(s_0) }_{\text{initial state distribution}}  \cdot \prod\limits_{t=0}^{T-1} \Big[ \underbrace{\pi (a_t \mid s_t, \theta)}_{\text{policy}} \cdot \underbrace{p(s_{t+1},r_t \mid s_t, a_t)}_{\text{transition fn.}} \Big]
$$

Before we had:

$$
\nabla_{\theta} \mathbb{E}_{x \sim p( x \mid s; \theta)} [r(x)] = \mathbb{E}_{x \sim p( x \mid s; \theta)} [r(x) \nabla_{\theta} \log p(x \mid s; \theta) ]
$$

Consider a stochastic generating process that gives us a trajectory of \\((s,a,r)\\) tuples:

$$
\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots, s_{T-1}, a_{T-1} r_{T-1}, s_T) 
$$

We want the expected reward over a trajectory...

$$
  \nabla_{\theta} \mathbb{E}_{\tau} [R(\tau)] = \mathbb{E}_{\tau} [ \underbrace{\nabla_{\theta} \log p(\tau \mid \theta)}_{\text{What is this?}} \underbrace{R(\tau)}_{\text{Reward of a trajectory}}  ]
$$

The reward over a trajectory is simple: \\(R(\tau) = \sum\limits_{t=0}^{T-1}r_t\\)

## The Probability of a Trajectory, Given a Policy

$$
\begin{align}
p(\tau \mid \theta) &= \underbrace{ \mu(s_0) }_{\text{initial state distribution}}  \cdot \prod\limits_{t=0}^{T-1} \Big[ \underbrace{\pi (a_t \mid s_t, \theta)}_{\text{policy}} \cdot \underbrace{p(s_{t+1},r_t \mid s_t, a_t)}_{\text{transition fn.}} \Big] & \text{Markov Property} \\
\mbox{log } p(\tau \mid \theta) &= \mbox{log } \mu(s_0) + \mbox{log } \prod\limits_{t=0}^{T-1} \Big[ \pi (a_t \mid s_t, \theta) \cdot p(s_{t+1},r_t \mid s_t, a_t) \Big] & \text{Log of product = sum of logs} \\
 &= \mbox{log } \mu(s_0) +  \sum\limits_{t=0}^{T-1} \mbox{log }\Big[ \pi (a_t \mid s_t, \theta) \cdot p(s_{t+1},r_t \mid s_t, a_t) \Big] & \text{Log of product = sum of logs} \\
 &= \mbox{log } \mu(s_0) +  \sum\limits_{t=0}^{T-1} \Big[ \mbox{log }\pi (a_t \mid s_t, \theta) + \mbox{log } p(s_{t+1},r_t \mid s_t, a_t) \Big] & \text{Log of product = sum of logs} \\
\end{align}
$$

## Maximizing Expected Return of a Trajectory
As we discussed earlier, in order to perform gradient ascent, we'll need to be able to compute:

$$
  \nabla_{\theta} \mathbb{E}_{\tau} [R(\tau)] = \mathbb{E}_{\tau} [ \underbrace{\nabla_{\theta} \log p(\tau \mid \theta)}_{\text{Derived Above}} \underbrace{R(\tau)}_{\text{Reward of a trajectory}}  ]
$$

We have just shown that the inner term is simply:

$$
\log p(\tau \mid \theta) = \mbox{log } \mu(s_0) +  \sum\limits_{t=0}^{T-1} \Big[ \mbox{log }\pi (a_t \mid s_t, \theta) + \mbox{log } p(s_{t+1},r_t \mid s_t, a_t) \Big]
$$

We need its gradient. Only one term depends upon \\(\theta\\) above, so all other terms fall out in the gradient:

$$
\nabla_{\theta} \mathbb{E}_{\tau} [R(\tau)] = \mathbb{E}_{\tau} \Big[ R(\tau) \nabla_{\theta} \sum\limits_{t=0}^{T-1} \mbox{log }\pi (a_t \mid s_t, \theta) \Big]
$$

This concludes the derivation of the Policy Gradient Theorem for entire trajectories.


## The REINFORCE Algorithm

The REINFORCE algorithm is a Monte-Carlo Policy-Gradient Method. Below we describe the episodic version:

+ Input: a differentiable policy parameterization \\(\pi(a \mid s, \theta)\\)
+ Initialize policy parameter \\(\theta \in \mathbb{R}^d\\)
+ Repeat forever:
    - Generate an episode \\(S_0\\), \\(A_0\\), \\(R_1\\), ... , \\(S_{T−1}\\), \\(A_{T−1}\\), \\(R_T\\) , following \\( \pi(\cdot \mid \cdot, \theta) \\)
    - For each step of the episode \\(t = 0, ... , T-1\\):
        - \\(G \leftarrow\\) return from step \\(t\\)
        - \\( \theta \leftarrow \theta + \alpha \gamma^t G \nabla_{\theta} \mbox{ ln }\pi(A_t \mid S_t, \theta) \\)

See page 271 of [1] for more details. Andrej Karpathy has a [very simple 130-line script here](https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5) that illustrates this in Python.


## REINFORCE With a Baseline

<!--- TODO: FIX THIS PART -->

In the section on geometric intuition, we discussed how:
Suppose we have a function \\(f(x)\\), such that \\(f(x) \geq 0 \forall x\\)
For every \\(x_i\\), the gradient estimator \\( \hat{g}_i\\) tries to push up on its density.

However, we really want to only push up on the density for better-than-average \\(x_i\\):

$$
\begin{align}
\nabla_{\theta} \mathbb{E}_x [f(x)] &= \nabla_{\theta} \mathbb{E}_x [f(x) -b] \\
&= \mathbb{E}_x \Big[ \nabla_{\theta}  \mbox{log } p(x \mid \theta) (f(x) -b) \Big]
\end{align}
$$

A near optimal choice of baseline \\(b\\) is always the mean return, \\( \mathbb{E}[f(x)]\\).                


## Actor Critic

Monte Carlo Policy Gradient has high variance. What if we used a critic to estimate the action-value function?

$$
Q_w(s,a) \approx Q^{\pi_\theta}(s,a)
$$

The Actor contains the policy and is the agent that makes decisions. The Critic makes no decisions or actions, but merely watches what the critic does and evaluates if it was good or bad.

## Actor-Critic with Approximate Policy Gradient

The critic says, “Hey, I think if you go in this direction, you can actually do better!”


In each step, move a little bit in the direction that the critic says is good or bad.


## Advantage Function

We remember the state-value function \\(V^{\pi}(s)\\), which tells us the goodness of a state.
We remember the state-action function \\(Q(s,a)\\).       
We want an advantage function \\(A^{\pi}(s,a)\\)  to tell us how much better is action \\(a\\) than what the policy \\(\pi\\) would have done otherwise:

$$
A^{\pi}(s,a) = Q(s,a) - V^{\pi}(s)
$$

Did things get better after one time step? Of course, a Critic could estimate both. But there’s a better way!

### TD Error as Advantage Function

Make the good trajectories more probable, make the bad trajectories less probable (high variance)

Monte Carlo
- unbiased but high variance
- conflates actions over whole trajectory

TD Learning
- introduces bias (treating guess as truth) but estimate have lower variance
- incorporates 1 step
- usually more efficient

For the true value function \\(V^{\pi_{\theta}}(s)\\), the TD error \\( \delta^{\pi_{\theta}}(s)\\)  

$$
\delta^{\pi_{\theta}}(s) = r + \gamma V^{\pi_{\theta}}(s^{\prime}) - V^{\pi_{\theta}}(s)
$$

is an unbiased estimate of the advantage function.

$$
\begin{aligned}
\mathbb{E}_{\pi_{\theta}} \Big[\delta^{\pi_{\theta}}(s) \mid s,a \Big] &= \mathbb{E}_{\pi_{\theta}} \Big[ r + \gamma V^{\pi_{\theta}}(s^{\prime}) - V^{\pi_{\theta}}(s) \Big] \\
= Q^{\pi_{\theta}}(s,a) - V^{\pi_{\theta}}(s) \\
= A^{\pi_{\theta}}(s,a)
\end{aligned}
$$

## TRPO: From 1st Order to 2nd Order

We don’t have access to the objective function in analytic form. However, we can estimate the gradient as the expected return of our policy by sampling trajectories. The gradient ascent update is correct if we make an infinitesimally small change.

A local approximation is only accurate closeby: 

<div class="fig figcenter fighighlight">
  <img src="/assets/trpo_local_approx.png" width="65%">
  <div class="figcaption">
    With higher function values f(x) to the right, the probability density p(x) will be pushed up by vectors with higher magnitude on the right.  Image source: John Schulman [2].
  </div>
</div>

Trust-region methods penalize large steps. Let’s call our surrogate objective \\(L_{\pi_{old}}(\pi)\\).
Let’s call our true objective \\( \eta(\pi)\\). We will add a penalty for moving \\( \theta \\) from \\( \theta_{old} \\) (KL Divergence between the old and new policy).

$$
\eta(\pi) \geq L_{\pi_{old}}(\pi) - C \cdot \max\limits_s KL \Big[ \pi_{old}(\cdot \mid s), \pi(\cdot \mid s) \Big]
$$

Monotonic improvement will be guaranteed.Since the max is not good for Monte Carlo estimation, a practical approximation is:

$$
\eta(\pi) \geq L_{\pi_{old}}(\pi) - C \cdot \mathbb{E}_{s\sim p_{old}} KL \Big[ \pi_{old}(\cdot \mid s), \pi(\cdot \mid s) \Big]
$$


### Second Order Optimization Method

What if we used a 2nd order optimization method instead of our vanilla 1st order optimization method?
We would compute the Hessian of the KL divergence, not of our objective 

Hessian computation is extremely expensive...( 100,000 x 100,000 )

### Truncated Natural Gradient Policy Algorithm

+ for iteration 1,2,..  do
    - Run policy for \\(T\\) timesteps or \\(N\\) trajectories
    - Estimate advantage function at all timesteps
    - Compute policy gradient \\(g\\)
    - Use Conjugate Gradient (with Hessian-vector products) to compute \\( H^{-1}g \\)
    - Update policy parameter \\( \theta = \theta_{old} + \alpha H^{-1}g \\)


### Trust Region Policy Optimization (TRPO)

The Truncated Natural Policy Gradient algorithm was an unconstrained problem:

$$
\eta(\pi) \geq L_{\pi_{old}}(\pi) - C \cdot \mathbb{E}_{s\sim p_{old}} KL \Big[ \pi_{old}(\cdot \mid s), \pi(\cdot \mid s) \Big]
$$


TRPO is a constrained problem (otherwise, algorithm is identical):

$$
\begin{aligned}
\mbox{max } & L_{\pi_{old}}(\pi) \\
\mbox{subject to } & \mathbb{E}_{s\sim p_{old}} KL \Big[ \pi_{old}(\cdot \mid s), \pi(\cdot \mid s) \Big] \leq \delta
\end{aligned}
$$


Easier to set hyper parameter \\(\delta\\) rather than \\(C\\). \\(\delta\\) can remain fixed over whole learning process... Simply rescale step by some scalar.


## References:

[1] Richard Sutton and Andrew Barto. Reinforcement Learning:
An Introduction. Chapter 13. [http://incompleteideas.net/book/bookdraft2017nov5.pdf](http://incompleteideas.net/book/bookdraft2017nov5.pdf).


[2] John Schulman. Deep Reinforcement Learning: Policy Gradients and Q-Learning.Bay Area Deep Learning School, September 24, 2016.

[3] Andrej Karpathy. Deep Reinforcement Learning: Pong from Pixels. May 31, 2016. [http://karpathy.github.io/2016/05/31/rl/](http://karpathy.github.io/2016/05/31/rl/).





