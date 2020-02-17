---
layout: post
comments: true
permalink: /stat-methods/
title:  "Statistical Methods"
excerpt: "central limit theorem (CLT), confidence intervals"
date:   2020-02-16 11:00:00
mathjax: true

---

Table of Contents:
- [CLT](#clt)
- [Confidence Intervals](#ci)



## CLT


Plot a histogram of all 5*100 data points.
1000 samples from a standard Normal distribution $N(0,1)$

For each sample (row), use the definition of the Chi-squared distribution to generate one Chi-squared random number with 5 degrees of freedom. Plot the histogram for these 100 Chi-squared random numbers.

\begin{equation}
\begin{aligned}
    Z_1 \sim N(0,1) \\
    \dots \\
    Z_n \sim N(0,1) \\
    X = \sum\limits_{i=1}^k Z_i^2 \sim \chi^2(k)
\end{aligned}
\end{equation}

Use R to directly generate 100 Chi-squared random numbers with 5 degrees of freedom. Plot using the histogram and compare the plots.

\textit{Find empirical} $\chi_{0.05,5}^2$ \textit{from the data using quantile as well as the exact value of qchisq function, and compare them. Note that this function works with the left tail probability.}

\textbf{We desire to find a value of $X=x, X \sim \chi^2(\nu)$ such that 5\% of the probability mass (area under curve) lies to the right of $X=x$ on its p.d.f. (CDF table for chi-squared measures from the right side). However, since the quantile function (similar to percentile) is measured from the left side, we use the 0.95 quantile or 95th percentile to find:}\\

1A.  (top left) 100 samples from simulated $\chi^2(5)$, using samples from $N(0,1)$. (top right) 100 samples from real $\chi^2(5)$.\\ \textbf{Comparison:} Samples from a simulated $\chi^2(\nu)$ vs. direct $\chi^2(\nu)$ samples look extremely similar.

Use these 100 normal random numbers and Chi-squared numbers you generated in Part a to generate 100 student’s t random numbers. Use R, to directly generate 100 t random numbers with 5 degrees of freedom. Plot the histogram and compare the plots.

\begin{equation}
\begin{aligned}
    & Z \sim N(0,1) \\
   &  V \sim \chi^2(\nu) \\
    & \frac{Z}{\sqrt{V/\nu}} \sim t(\nu)
\end{aligned}
\end{equation}


\textit{Plot a histogram of all n*100 data points.}

\subsubsection{}
\textit{Compute the average of each sample (you should have 100 average values) and plot a histogram for Xbar values.}

\subsubsection{}
\textit{Repeat this procedure for various n values, namely (5, 10, 30, 50, 100), and find the minimum n whose corresponding Xbar histogram looks normal.}

\textbf{The more samples we draw and average over, the more normal the distribution looks, which follows CLT. In addition, the CLT says that your variance is inversely proportional to \# samples. This is confirmed in our plots because  variance is definitely shrinking as \# samples grows.}


```
import matplotlib.pyplot as plt
import numpy as np
import pdb

def sec_1a(x_normal, x_chi_sim, x_chi_direct):
	""" """
	cum_sim = np.quantile(x_chi_sim, q=0.95)
	cum_direct = np.quantile(x_chi_direct, q=0.95)
	print(f'Cum. simulated vs. direct: {cum_sim} vs. {cum_direct}')

	plt.subplot(1,3,1)
	plt.hist(x_normal.squeeze(), range=[-5,5])
	plt.title('1A Normal')

	plt.subplot(1,3,2)
	plt.hist(x_chi_sim, range=[-1,20])
	plt.title('1A Chi-squared Sim.')

	plt.subplot(1,3,3)
	plt.hist(x_chi_direct, range=[-1,20])
	plt.title('1A Chi-squared Direct')
	plt.show()

def sec_1b(x_t_direct, x_t_sim):
	""" """
	cum_sim = np.quantile(x_t_sim, q=0.95)
	cum_direct = np.quantile(x_t_direct, q=0.95)
	print(f'Cum. simulated vs. direct: {cum_sim} vs. {cum_direct}')

	plt.subplot(1,2,1)
	plt.hist(x_t_sim, bins=20, range=[-8,8])
	plt.title('1B student t Sim.')

	plt.subplot(1,2,2)
	plt.hist(x_t_direct, bins=20, range=[-8,8])
	plt.title('1B student t Direct')
	plt.show()


def sec_1c():
	""" """
	x_chi_1df = np.random.chisquare(df=1, size=100)
	x_chi_5df = np.random.chisquare(df=5, size=100)
	x_F_sim = (x_chi_1df/1) / (x_chi_5df/5)

	x_F_direct = np.random.f(dfnum=1, dfden=5, size=100)

	cum_sim = np.quantile(x_F_sim, q=0.95)
	cum_direct = np.quantile(x_F_direct, q=0.95)
	print(f'Cum. simulated vs. direct: {cum_sim} vs. {cum_direct}')

	plt.subplot(1,2,1)
	plt.hist(x_F_sim, bins=20, range=[-1,20])
	plt.title('1C F Sim.')

	plt.subplot(1,2,2)
	plt.hist(x_F_direct, bins=20, range=[-1,20])
	plt.title('1C F Direct')
	plt.show()


def part1():
	""" """
	x_normal = np.random.normal(loc=0.0, scale=1.0, size=(100,5))
	x_chi_sim = (x_normal**2).sum(axis=1)
	x_chi_direct = np.random.chisquare(df=5, size=100)
	sec_1a(x_normal, x_chi_sim, x_chi_direct)

	x_normal = np.random.normal(loc=0.0, scale=1.0, size=100)
	x_t_direct = np.random.standard_t(df=5, size=100)
	x_t_sim = x_normal / np.sqrt(x_chi_sim / 5)
	sec_1b(x_t_direct, x_t_sim)

	sec_1c()


def sec_2a():
	""" """
	exp_lambda = 0.2
	n_values = [5,10,30,50,100]
	for i, n in zip(range(1,6), n_values):
		x_exp = np.random.exponential(scale=1/exp_lambda, size=(100,n))
		x_exp_avg = np.mean(x_exp, axis=1)

		plt.subplot(2,len(n_values),i)
		plt.hist(x_exp_avg, bins=20, range=[-1,10])
		plt.title(f'2A Exp, n={n}')

		plt.subplot(2,len(n_values),len(n_values) + i)
		plt.hist(x_exp, bins=20, range=[-1,10])
		plt.title(f'2A Exp, n={n}')

	plt.show()



def sec_2b():
	""" """
	n_values = [5,10,30,50,100]
	for i, n in zip(range(1,6), n_values):

		x_gamma = np.random.gamma(shape=10, scale=5, size=(100,n))
		x_gamma_avg = np.mean(x_gamma, axis=1)

		plt.subplot(2,len(n_values),i)
		plt.hist(x_gamma_avg, bins=20, range=[25,75])
		plt.title(f'2B Gamma, n={n}')

		plt.subplot(2,len(n_values),len(n_values) + i)
		plt.hist(x_gamma.squeeze(), bins=20, range=[0,100])
		plt.title(f'2B Gamma, n={n}')

	plt.show()


if __name__ == '__main__':
	#part1()

	#sec_2a()
	sec_2b()
```



## Confidence Intervals

Understanding the concept of confidence interval: Using R, generate 1000 samples of 5 standard normal random numbers.

\begin{enumerate}[label=\emph{\alph*})]
\item \textit{For each sample (row), find a 95\% confidence interval.}

We should think of this as: take 5 samples, and repeat this experiment 1000 times.

Since our samples are from $Z \sim N(\mu=0,\sigma^2=1)$, we can compute the confidence interval as:

\begin{equation}
\begin{aligned}
\overline{X} - Z_{\alpha/2} \times \frac{\sigma}{\sqrt{n}} &\leq \mu \leq \overline{X} + Z_{\alpha/2} \times \frac{\sigma}{\sqrt{n}} \\
\overline{X} - Z_{\alpha/2} \times
\frac{1}{\sqrt{n}} &\leq \mu \leq \overline{X} + Z_{\alpha/2} \times \frac{1}{\sqrt{n}} \\
\overline{X} - Z_{\alpha/2} \times
\frac{1}{\sqrt{5}} &\leq \mu \leq \overline{X} + Z_{\alpha/2} \times \frac{1}{\sqrt{5}} \\
\overline{X} - Z_{\alpha/2} \times
0.447 &\leq \mu \leq \overline{X} + Z_{\alpha/2} \times 0.447 \\
\overline{X} - 1.96 \times
0.447 &\leq \mu \leq \overline{X} + 1.96 \times 0.447 \\
\overline{X} - 0.877 &\leq \mu \leq \overline{X} + 0.877 \\
\end{aligned}
\end{equation}



\item  \textit{What is the percentage of intervals that cover the true mean? What is your conclusion?}

On average, how often does the population parameter of interest (here, $\mu=0$), belong to that interval?

We ran 3 separate random trials of our experiment.\\
Percentages of intervals that cover true mean, $\mu=0$: \\
962 / 1000 = 96.2\%, \\
966 / 1000 = 96.6\%, \\
940 / 1000 = 94.0\%, 

Our conclusion is that (on average over several experiments), the empirical confidence interval is approximately equal to the analytical confidence interval.





```python
simdata = np.random.randn(1000*5)
X = simdata.reshape(1000,5)
Xsums = np.sum(X,axis=1)
Xsums /= 5
np.logical_and( (Xsums + 0.877 > 0 ), (Xsums - 0.877 < 0 ) ).sum()
```

We also verify the percent point function (inverse of cdf --  percentiles):
`scipy.stats.norm.ppf(0.025)` = -1.96
`scipy.stats.norm.ppf(0.975)` = 1.96