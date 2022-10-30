---
layout: post
title:  "Weighted K-Means as Weighted Pairwise Distance"
date:   2022-10-28 01:01:01 -0400
categories: jekyll update
usemathjax: true
---
I could not find a proof of this online, so I decided to quickly write this up in case it helps anybody else with research.

Let $$x_i \in \mathbb{R}^d$$ for $$i=1,...,n$$. Each sample $$x_i$$ has a weight $$\alpha_i$$, $$i=1,...,n$$. The weighted mean for a cluster is $$\mu = \frac{\sum_{i=1}^n \alpha_i x_i}{\sum_{i=1}^n \alpha_i}$$.

As per Scikit-Learn's documentation, a weight is treated like multiple of the same sample. So, the goal is:

$$
    \min \, \sum_i \alpha_i ||x_i - \mu ||^2
$$

This is equivalent to minimizing the weighted pairwise distances:

$$
    \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j ||x_i - x_j||^2
$$

As demonstrated below:

$$
\begin{align*}
    \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j ||x_i - x_j||^2
    &= \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j \left(||x_i||^2 - 2x_i^Tx_j+||x_j||^2\right)\\
    &= \left(\sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j ||x_i||^2\right) - 2\left(\sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j x_i^Tx_j\right) + \left(\sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j ||x_j||^2\right) \\
    &= 2\left(\sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j ||x_i||^2\right) - 2\left(\sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j x_i^Tx_j\right) \\
    &= 2\left(\sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j ||x_i||^2\right) - 4\left(\sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j x_i^Tx_j\right) + 2\left(\sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j x_i^Tx_j\right)
\end{align*}
$$

A few intermediate identities:

$$
\begin{align*}
    ||\mu||^2
    &= \frac{||\sum_{i=1}^n \alpha_i x_i||^2}{(\sum_{i=1}^n\alpha_i)^2} \\
    &= \frac{\sum_{i=1}^n\sum_{j=1}^n \alpha_i\alpha_jx_i^Tx_j}{(\sum_{i=1}^n\alpha_i)^2}\\
    ||\mu||^2\left(\sum_{i=1}^n\alpha_i\right)^2 &= \sum_{i=1}^n\sum_{j=1}^n \alpha_i\alpha_jx_i^Tx_j
\end{align*}
$$

Next,

$$
\begin{align*}
    4\sum_{i=1}^n\sum_{j=1}^n \alpha_i \alpha_j x_i^Tx_j
    &= 4\sum_{i=1}^n\left(\alpha_i x_i^T \sum_{j=1}^n \alpha_j x_j\right) \\
    &= 4\left(\sum_j\alpha_j\right)\sum_{i=1}^n\left(\alpha_i x_i^T \frac{\sum_{j=1}^n \alpha_j x_j}{\sum_j\alpha_j}\right) \\
    &= 4\left(\sum_j\alpha_j\right)\sum_{i=1}^n\alpha_i x_i^T\mu
\end{align*}
$$

Putting this together,

$$
\begin{align*}
    \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j ||x_i - x_j||^2
    &= 2\left(\sum_j\alpha_j\right)\sum_{i=1}^n\alpha_i ||x_i||^2 - 4\left(\sum_j\alpha_j\right)\sum_{i=1}^n\alpha_i x_i^T\mu + 2||\mu||^2\left(\sum_{i=1}^n\alpha_i\right)^2\\
    &= 2\left(\sum_j\alpha_j\right)\sum_{i=1}^n\alpha_i ||x_i||^2 - 4\left(\sum_j\alpha_j\right)\sum_{i=1}^n\alpha_i x_i^T\mu + 2\left(\sum_{j=1}^n\alpha_j\right)\sum_{i=1}^n\alpha_i||\mu||^2\\
    &= 2\left(\sum_j\alpha_j\right)\left(\sum_{i=1}^n \alpha_i||x_i||^2 - 2\alpha_ix_i^T\mu+\alpha_i||\mu||^2\right) \\
    &= 2\left(\sum_j\alpha_j\right)\left(\sum_{i=1}^n \alpha_i(||x_i||^2 - 2x_i^T\mu+||\mu||^2)\right) \\
    &= 2\left(\sum_j\alpha_j\right)\left(\sum_{i=1}^n \alpha_i||x_i-\mu||^2\right) \\
    \frac{\sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j ||x_i - x_j||^2}{2\sum_j\alpha_j} &= \sum_{i=1}^n \alpha_i||x_i-\mu||^2
\end{align*}
$$

Finally, we can remove the half coeffient:

$$
  \frac{\sum_{i < j} \alpha_i \alpha_j ||x_i - x_j||^2}{\sum_j\alpha_j} = \sum_{i=1}^n \alpha_i||x_i-\mu||^2
$$
