---
layout: post
title:  "Understanding Softmax Cross Entropy"
permalink: /cross-entropy/
excerpt: "information theory, maximum likelihood"
mathjax: true
date:   2019-06-18 11:00:00

---
Table of Contents:
- [Why do we need residual connections?](#need-for-residual)


## Measures of Information

Common measures of information include entropy, mutual information, joint entropy, conditional entropy, and the s Kullback-Leibler (KL) divergence (also known as relative entropy).




## Surprise


Let $$U$$ be a discrete random variable, taking values in $${\mathcal U} = \{ u_1, u_2, \dots, u_M\}$$. The *Surprise* Function $$S(u)$$ is defined as

$$
S(u) = \log_2 \Big( \frac{1}{p(U=u)} \Big) = \log_2 \Big( \frac{1}{p(u)} \Big)
$$

To make that more concrete, 

```python
surprise = lambda x: np.log2(1./x)
p_u = np.linspace(0,1,11)
```
We obtain probability values $$p(u)$$, from small to large:
```
p_u
array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ])
```
and surprise values that decrease as $$p(u)$$ increases:
```
array([ inf, 3.32, 2.32, 1.74, 1.32, 1.  , 0.74, 0.51, 0.32, 0.15, 0.  ])
```
We can plot the values:
```python
import matplotlib.pyplot as plt
plt.plot(p_u,S,'r')
plt.xlabel(r'$p(u)$')
plt.ylabel(r'Surprise $S(u)$')
plt.show()
```

<div class="fig figcenter fighighlight">
  <img src="/assets/surprise_function.png" width="45%">
  <div class="figcaption">
    Less probable events generate higher "surprise" values (they surprise us more than very likely outcomes).
  </div>
</div>

## Entropy

The entropy of $$U$$ is simply the *expected* Surprise. More formally,


$$
\begin{aligned}
H(U) &= \mathbb{E}[S(U)] \\
     &= \mathbb{E}\Big[\log_2\big(\frac{1}{p(U)}\big)\Big] \\
     &= \mathbb{E}\Big[\log_2\big(p(U)^{-1}\big)\Big] \\
     &= \mathbb{E}\Big[-\log_2 p(U)\Big] & \text{because } \log x^{k} = k \log x \\
     &= - \sum\limits_u p(u) \log_2 p(u) & \text{by definition of expectation}
\end{aligned}
$$

Thus, entropy measures the amount of surprise or randomness in the random variable [1].

Consider the entropy of 3 simple distributions over $${\mathcal U} = \{ u_1, u_2, u_3, u_4\}$$, and their probability masses $$[p(U=u_1),p(U=u_2),p(U=u_3),p(U=u_4)]$$.

```python
cross_entropy = lambda x: -np.sum(np.log2(x)*x)

p1 = np.array([0.97, 0.01, 0.01, 0.01])
surprise(p1)
array([0.04, 6.64, 6.64, 6.64])
```
The entropy of p1 is 0.24, since although we get some high surprise values, they have very little probability mass.

```python
p2 = np.array([0.5,0.3, 0.19, 0.01])
surprise(p2)
array([1.  , 1.74, 2.4 , 6.64])
```
The entropy of p2 is 1.54, and once again, although we get one high surprise value, it has almost no probability mass: $$p(U=u_4)=0.01$$.

```python
p3 = np.array([0.25, 0.25, 0.25, 0.25])
surprise(p3)
array([2., 2., 2., 2.])
```
Finally, the entropy of p3 is 2.0. We get the most average surprise from p3 (a uniform distribution).


## KL Divergence (Relative Entropy)

The Kullback-Leibler (KL) divergence, also known as relative entropy, is defined as




<a name='need-for-residual'></a>
## Cross Entropy


$$
H(p,q) = - \sum_x p(x) \log q(x)
$$

$$
H(p,q) = H(p) + D_{KL}(p||q)
$$


In PyTorch, the negative log likelihood loss ([NLL loss](https://pytorch.org/docs/stable/nn.html#torch.nn.NLLLoss)) can be used for multinomial classification, and expects to receive log-probabilities for each class.

Consider an empirical output distribution `[0.2, 0.6, 0.2]` and a target distribution `[0,1,0]`, in a one-hot format. The index of the target class is simply `1`, which is the format PyTorch expects for the target. `NLLLoss()` expects log-probabilities:

```python
nll = torch.nn.NLLLoss()
x = torch.tensor([[0.2, 0.6, 0.2]])
x = np.log(x)
y = torch.tensor([1])
loss = nll(x,y)
print(loss)
	tensor(0.5108, dtype=torch.float64)
```

Our goal is to minimize this quantity. We can do so by making the target and empirical distribution match more closely. When the two distributions come closer and closer, the loss (divergence between them) decreases:

```python
for dist in [ [0.3, 0.4, 0.3], [0.2, 0.6, 0.2], [0.1, 0.8, 0.1], [0,1.,0] ]:
	nll = torch.nn.NLLLoss()
	x = torch.tensor([np.log(dist)])
	y = torch.tensor([1])
	loss = nll(x,y)
	print(loss)
```
In this case, `0.9163, 0.5108, 0.2231, 0.` is printed, meaning the loss drops as the distributions align, until it finally reaches zero when the distributions are identical.



(`CrossEntropyLoss` incorporates `LogSoftmax` inside of it).


## Logistic Regression

## Softmax Regression


## The Softmax Operation

```python
soft = nn.Softmax(dim=1)
x = torch.tensor([[0,1,2],[0.8, 0.4, 0.4]])
x = soft(x)
print(x)
```

The first row can identically be computed by:
```python
x = np.array([0,1,2])
x = np.exp(x)
x = x / np.sum(x)
print(x)
```

Consider the Negative Log Likelihood (NLL) loss criterion.


## References

[1] Tsachy Weissman. Information Theory (EE 376, Stanford) Course Lectures. [PDF](http://web.stanford.edu/class/ee376a/files/2017-18/lecture_3.pdf).

[2]















