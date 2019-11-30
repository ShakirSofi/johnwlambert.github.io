---
layout: post
title:  "Metric Learning"
permalink: /metric-learning/
excerpt: "contrastive loss, triplet loss, angular softmax"
mathjax: true
date:   2019-06-18 11:00:00

---
Table of Contents:
- [Why do we need residual connections?](#need-for-residual)


## A hyperplane is a linear classifier

A hyperplane is a plane in n-dimensions that can split a space into two halfspaces. For the sake of simplicity, consider a hyperplane in 2-dimensional space (a line).

A common and intuitive representation of a line in two-dimensions, $$\mathbb{R}^2$$, is $$y=mx+b$$, where $$y=x_2$$ and  $$x=x_1$$. Here $$m$$ represents slope, and $$b$$ represents a y-intercept. This representation is a poor choice, however, because we cannot represent vertical lines, e.g. $$ x = c $$, with this expression. A vertical line equation would have to be something like $$y = mx$$, where $$m \rightarrow \infty$$. Thus, this isn't a general equation of a line.

Over $$\mathbb{R}^2$$, a better representation is 

$$
w_0 + w_1 x_1 + w_2 x_2 = 0
$$

where $$w_0, w_1, w_2$$ are weights ("parameters"), and $$w_0$$ is often considered the "bias" term, denoting the distance from the origin.  When $$w_2 =0$$, we have a vertical line, and when $$w_1=0$$, we have a horizontal line. We can always convert back to our intuitive representation $$y=mx+b$$ by simple algebra. For example, consider the hyperplane

$$
\begin{array}{ll}
W = \begin{bmatrix} w_1 \\ w_2 \end{bmatrix}^T = \begin{bmatrix} 2 \\ 1 \end{bmatrix}^T, & w_0 = b = -2
\end{array}
$$

By algebra:

$$
\begin{aligned}

w_0 + w_1 x_1 + w_2 x_2 = 0 \\
w_0 + w_1 x + w_2 y = 0 & \mbox{ (let } y = x_2, x = x_1) \\
w_2 y =  -w_1 x - w_0 \\
y = -\frac{w_1}{w_2}x + -\frac{w_0}{w_2} \\
y = -\frac{2}{1}x + -\frac{-2}{1} &  \mbox{ (let } w_0 = -2, w_1 = 2, w_2 = 1) \\
y = -2x + 2
\end{aligned}
$$


```python
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-3,3,100)
y = -2*x + 2
plt.scatter(x,y,10,marker='.', color='r')
plt.xlim([-3,3])
plt.ylim([-6,6])
plt.show()
x.shape
(100,)
np.allclose(-2 + 2 * x + 1 * y, np.zeros(100) )
True
```


http://cs231n.github.io/linear-classify/

https://cedar.buffalo.edu/~srihari/CSE574/Chap4/4.1%20DiscFns.pdf

http://grzegorz.chrupala.me/papers/ml4nlp/linear-classifiers.pdf




## Angular Softmax

SphereFace



## Hinge Loss


max-margin


## Contrastive Loss

The contrastive loss was proposed by Hadsell, Chopra, and LeCun in [2] for dimensionality reduction of high-dimensional points.

Given a set of input vectors $$ \mathcal{I} = \{ \vec{X}_1, \dots, \vec{X}_P \}$$, where $$\vec{X}_i \in \mathbb{R}^D, \forall i = 1, \dots, n$$, find a parametric function $$G_W: \mathbb{R}^D \rightarrow \mathbb{R}^d$$ with $$d ≪ D$$,

Unlike conventional learning systems where the loss function is a sum
over samples, the contrastive loss function here runs over pairs of samples [2].

Here is the loss formulation, as described in [2]: Let $$\vec{X}_1, \vec{X}_2 \in I$$ be a pair of input vectors shown to the system. Let $$Y$$ be a binary label assigned to this pair. $$Y = 0$$ if $$\vec{X}_1$$ and $$\vec{X}_2$$ are deemed similar (i.e. from the same semantic class), and $$Y = 1$$ if they are deemed dissimilar (i.e. from different semantic classes). Define the parameterized distance function to be learned $$D_W$$ between $$\vec{X}_1, \vec{X}_2$$ as the Euclidean distance between the outputs of $$G_W$$. That is,
$$D_W (\vec{X}_1, \vec{X}_2) = \|G_W (\vec{X}_1) − G_W (\vec{X}_2)\|_2$$ (1)
To shorten notation, $$D_W (\vec{X}_1, \vec{X}_2$$) is written $$D_W$$. Then the loss function in its most general form is

$$
L(W) = \sum\limits_{i=1}^P L\Big(W,(Y, \vec{X}_1, \vec{X}_2)^i\Big)
$$

$$
L\Big(W,(Y, \vec{X}_1, \vec{X}_2)^i\Big) = (1 − Y )L_S (D_W^i)+ Y L_D (D_W^i)
$$


where $$(Y, \vec{X}_1, \vec{X}_2)^i$$ is the $$i$$-th labeled sample pair, $$L_S$$ is the partial loss function for a pair of similar points, $$L_D$$ the
partial loss function for a pair of dissimilar points, and $$P$$ the number of training pairs (which may be as large as the square of the number of samples.


The exact, full loss function is:

$$
L(W, Y, \vec{X}_1, \vec{X}_2) = (1 − Y ) \frac{1}{2} (D_W )^2 + (Y )
\frac{1}{2} \{ \mbox{max}(0, m − D_W )\}
$$


Consider the loss function $$L_S(W, \vec{X}_1, \vec{X}_2)$$ associated
with similar pairs. The data points in each pair should pulled together:

$$
L_S(W, \vec{X}_1, \vec{X}_2) = \frac{1}{2}(D_W )^2
$$

Now consider the partial loss function $$L_D$$ associated with dissimilar pairs. The data points in each pair should be pushed apart:

$$
L_D(W, \vec{X}_1, \vec{X}_2) = \frac{1}{2}(\mbox{max}\{0, m − D_W \})
$$

We often use a a siamese network architecture, which consists of two
copies of the function $$G_W$$ which share the same set of parameters $$W$$, and a cost module. A loss module whose input is the output of this architecture is placed on top of it.



Compute contrastive loss with push term for different class pairs and pull term for same-class pairs. Push term says if pair elements come from separate classes, push embeddings apart. Pull term says if pair elements come from the same class, pull embeddings together. Sum losses as contrastive loss*

We could implement such a contrastive loss in Pytorch as follows



```python

def contrastive_loss(y_c: torch.Tensor, pred_dists: torch.Tensor, margin: int = 1) -> torch.Tensor:
    """
        Compute the similarities in the separation loss by 
        computing average pairwise similarities between points
        in the embedding space.

		element-wise square, element-wise maximum of two tensors.

        Args:
        -   y_c: Indicates if pairs share the same semantic class label or not
        -   pred_dists: Distances in the embeddding space between pairs. 

        Returns:
        -   tensor representing contrastive loss value.
    """
    N = pred_dists.shape[0]

    # corresponds to "d" in the paper. If same class, pull together.
    # Zero loss if all same-class examples have zero distance between them.
    pull_losses = y_c * torch.pow(pred_dists, 2)
    # corresponds to "k" in the paper. If different class, push apart more than margin
    # if semantically different examples have distances are in [0,margin], then there WILL be loss
    clamped_dists = torch.max(margin - pred_dists, torch.zeros(N) )
    push_losses = (1 - y_c) * torch.pow(clamped_dists, 2)
    return torch.mean(pull_losses + push_losses)

```

Consider 5 pairs of points. For two of these pairs, the tuple of points $$(x_A,x_B)$$ pairs having elements belonging to the same class, and the rest have elements classes. Suppose the 2 data points that belong to the same semantic class have 0 distance between themselves, and all other points are at distance 1.1

```python

def test_contrastive_loss1():
    """
    Should be no loss here (zero from pull term, and zero from push term)
    """
    # which pairs share the same semantic class label
    y_c = torch.tensor([ 1., 0., 0., 0., 1.], dtype=torch.float32)

    # distances between pairs
    pred_dists = torch.tensor([0, 1.1, 1.1, 1.1, 0], dtype=torch.float32)

    loss = contrastive_loss(y_c, pred_dists)
    gt_loss = torch.tensor([0])

    assert torch.allclose(loss, gt_loss)
```

Now consider if the 
```python
def test_contrastive_loss2():
    """ 
    There should be more loss here (coming only from push term)
    """
    # which pairs share the same semantic class label
    y_c = torch.tensor([ 1., 0., 0., 0., 1.], dtype=torch.float32)

    # distances between pairs
    pred_dists = torch.tensor([0, 0.2, 0.3, 0.1, 0], dtype=torch.float32)

    loss = contrastive_loss(y_c, pred_dists)
    gt_loss = torch.tensor([0.3880])

    assert torch.allclose(loss, gt_loss, atol=1e-3)
```

```python
def test_contrastive_loss3():
    """
    There should be the most loss here (some from pull term, and some from push term also)
    """
    # which pairs share the same semantic class label
    y_c = torch.tensor([ 1., 0., 0., 0., 1.], dtype=torch.float32)

    # distances between pairs
    pred_dists = torch.tensor([2.0, 0.2, 0.3, 0.1, 4.0], dtype=torch.float32)

    loss = contrastive_loss(y_c, pred_dists)
    gt_loss = torch.tensor([4.3880])

    assert torch.allclose(loss, gt_loss, atol=1e-3)

```


## Triplet Loss


https://github.com/lugiavn/generalization-dml

Hard negative mining


## Which layer to fine-tune?





## Additive Margin Softmax



[1] Weiyang Liu, Yandong Wen, Zhiding Yu, Ming Li, Bhiksha Raj, Le Song. SphereFace: Deep Hypersphere Embedding for Face Recognition. [arXiv](https://arxiv.org/abs/1704.08063).

[2] Raia Hadsell, Sumit Chopra, Yann LeCun. Dimensionality Reduction by Learning an Invariant Mapping. CVPR 2006. [PDF](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf).

[3]