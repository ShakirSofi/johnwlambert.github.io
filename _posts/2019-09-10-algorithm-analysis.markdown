---
layout: post
title:  "Algorithm Analysis"
permalink: /algorithm-analysis/
excerpt: "Master Theorem, Divide and Conquer"
mathjax: true
date:   2019-09-10 11:00:00
mathjax: true

---
Table of Contents:
- [Master Theorem](#state-estimation)

<a name='state-estimation'></a>


## Divide and Conquer Algorithms

### Recurrences

Black box for solving recurrence, get an upper bound on the algorithm's running time

Only relevant when all of the subproblems have exactly the same size (recursing on 1/3 of the array, and then 1/2 of the array), would not be suitable

Recurrence Format:
1. Base case $$T(n) \leq$$ a constant for all sufficiently small $$n$$

2. For all larger $$n$$:
$$T(n) \leq a T (\frac{n}{b}) + cn^d$$

where 

$$a$$ is number of recursive calls $$(\geq 1)$$
$$b =$$ input size shrink factor $$(>1)$$
$$d =$$ exponent in running time of combine step

[a,b,d independent of n]

### Formal Statement of Master Theorem

$$
T(n) \begin{cases} O(n^d \log n) & \mbox{ if } a = b^d \mbox{ (Case 1)} \\
O(n^d) & \mbox{ if } a < b^d \mbox{ (Case 2)} \\
O(n^{\log_b a}) & \mbox{ if } a > b^d \mbox{ (Case 3)}
\end{cases}
$$

At level j in the recursion tree, there are $$a^j$$ problems, and each subproblem is of size $$n / b^j$$.

Get a bound on the work done at level $$j$$

$$
\begin{aligned}
& \leq \underbrace{a^j}_{\text{# of level-j subproblems}} \cdot c \Bigg[ \underbrace{ \frac{n}{b^j}}_{\text{size of each level-j subproblem}} \Bigg]^d \\
&= cn^d \cdot [\frac{a}{b^d}]^j
\end{aligned}
$$

Now, summing over all levels $$j=0,1,2,\dots, \log_b n$$:

$$ \mbox{Total Work} \leq cn^d \sum\limits_{j=0}^{\log_b n} [\frac{a}{b^d}]^j$$


Surprisingly, 
Forces of good:
$$b^d = $$ rate of work shrinkage (RWS) (per sub-problem)

Forces of evil:
$$a = $$ rate of subproblem proliferation (RSP), function of $$J

Why is the rate of work-shrinkage not just $$b$$? Why is it $$b^d$$? This is because we care about how much work goes down per sub-problem. Linear -> down to half, Quadratic -> down to quarter.

1. If RSP < RWS, then the amount of work is *decreasing* with the recursion level $$j$$.
2. If RSP > RWS, then the amount of work is *increasing* with the recursion level $$j$$.
3. If RSP and RWS are equal, then the amount of work is *the same* at every recursion level $$j$$.

1. RSP > RWS means most work at the leaves! (leaves dominant), get $$O(\# Leaves)$$
2. RSP < RWS means less work at each level (most work at the root!) Might expect $$O(n^d)$$
3. In the tie, there is same amount of work at each level (like MergeSort). Logarithmic number of levels, do $$n^d$$ work at each level. So we would expect $$O(n^d \log n)$$

In practice, we see that in case #1, we did not get $$O(\# Leaves)$$, but rather $$O(n^{\log_b a})$$. Turns out, these are identical, since there are $$a^{log_b n}$$ leaves in the recursion tree. Since $$a$$ is the branching factor, this process continues until we get to the leaves (number of times we multiply by $$a$$).


In the Equilibrium, $$a=b^d$$:
$$ 
\begin{aligned}
\mbox{Total Work} \leq cn^d \sum\limits_{j=0}^{\log_b n} [\frac{a}{b^d}]^j \\
\mbox{Total Work} \leq cn^d \sum\limits_{j=0}^{\log_b n} [1]^j \\
\mbox{Total Work} \leq cn^d (\log_b n + 1)
\end{aligned}
$$

It turns out that by the Power Rule of logarithms, i.e. $$ y \log_a x = \log_a x^y $$, (see a proof here of this rule): [link](https://www.khanacademy.org/math/algebra2/x2ec2f6f830c9fb89:logs/x2ec2f6f830c9fb89:log-prop/a/justifying-the-logarithm-properties)



$$
\begin{aligned}
a^{\log_b n} &= n^{\log_b a} \\
(\log_b n)(\log_b a) &= (\log_b a)(\log_b n)
\end{aligned}
$$

While the left side is more intuitive, the right side is easier to apply and evaluate.

### Example: MergeSort




### Closest Pair


### Fast Fourier Transform



## Dynamic Programming

Least Common Subsequence


## References

[1] Tim Roughgarden, Lectures.

