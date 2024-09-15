# SDE 微分方程角度

###简介 

前面我们介绍DDPM、NCSN，分别从噪声预测（理论层面是由 ELBO 推导而来）和 Score 的角度解释了扩散模型的扩散和生成过程。而[这篇论文](https://arxiv.org/abs/2011.13456)，则提供了一个新的视角：随机微分方程。

一般而言，我们加噪和去噪，其实都是随着时间一点一点进行的。比如在 DDPM 中，不同时间步图片加噪造成的图像破坏是不同的，而生成过程则相反，随着时间的“后退”，图像越来越清晰。这个过程是定义在离散时间步上的，但我们可以让它连续起来，构造一个图片 $x$（或其它信号）关于时间 $t$ 的一个函数。这个函数有前向和反向过程，这就分别描述了加噪和去噪的过程。

举例来说，假设你知道一个微分方程是 $dx=2t dt$，而且你知道它在 $t=0$ 时，$x=0$（很明显这个方程是 $x=t^2$，但你假装不知道），那么你其实可以用数值方法去求解 $t$ 在某个点的值。具体来说，假如我们想知道 $t=1$ 时 $x$ 是多少，那我们从 $t=0$ 处出发，取一个很小的 $t$ 间隔 $\Delta t$，通过上述微分方程就可以算出一个近似的 $\Delta x$，作为 $x$ 经过这一小段 $t$ 的变化后产生的变化量。通过这个变化量，你可以更新 $x$，一步一步，直到更新到 $t=1$。上述方法就是数值求解微分方程的 Euler Method。当然，这种方式积累误差会比较大，尤其是对于像 $x=t^2$ 这样的方程（你可以自己写个程序看看误差多大）。

对应到扩散模型上，我们把 $x$ 想象成一个张量（而不是上面例子的标量），随着时间一步步求解 $x$ 的变化量，就可以将这个过程类比到对图片或其它数据加噪或去噪过程上。比如说，假如我们要生成图片，在 $t=0$ 时刻，我们知道 $x$ 是某个随机噪声，那么我们根据一个微分方程，一步步求解一小段时间变化下 $x$ 的变化量，从随机噪声开始更新 $x$，最终你也能得到一个生成图片。

此外，前面例子中并没有随机成分，它的方程轨迹是一条确定的线路。在扩散模型生成过程中，无论是 DDPM 还是 NCSN，其实都是有随机项的，这样采样出来的图片才足够多样。加入随机项的微分方程，就叫随机微分方程（Stochastic Differential Equation）；而前面例子中的，则称为常微分方程（Ordinal Differential Equation）。也即 SDE 和 ODE。

### 微分方程的形式

回想 DDPM 的加噪公式 $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1 - \bar\alpha_t}\epsilon$，以及 NCSN 中，$x_t = x_0 + \sigma_t \epsilon$，可以发现，公式中一项均与 $x$ 相关，而另一项均是一项高斯噪声 $\epsilon$。这两个例子中，我们可以发现，其实 $x$ 在不同时刻均服从一个高斯分布，而其均值和方差在不同时刻是变化的。我们可以将随机微分方程设计成这样：
$$
dx=f(x,t)dt + g(t)dw
$$
其中 $w$ 是一个 [Wiener Process](https://mathworld.wolfram.com/WienerProcess.html)（为什么用 Wiener Process？求微分时需要用到两个时间上相邻的 $x$ 相减，两个高斯噪声之差就是 Wiener Process）。式中的两项分别叫 drift 和 diffusion，前者代表了确定的轨迹，后者则加入了随机项，以保证样本多样性。

加噪过程如上式描述，那去噪过程呢？反向时间的微分方程形式如下（具体原理可见[这篇论文](https://www.sciencedirect.com/science/article/pii/0304414982900515)）：
$$
dx=[f(x,t)-g(t)^2\nabla_x log p_t(x)]dt + g(t) d\bar w
$$

有了上面式子，只要我们知道 $f(x,t)$ 和 $g(t)$ 的具体形式，再加上我们用神经网络预测 Score（反向过程涉及到了 score），我们就可以进行扩散和生成了。

### 以 NCSN 为例

回忆一下，NCSN 中，加噪公式是这样：$x_t = x_0 + \sigma_t \epsilon$，这不是一个马尔可夫过程。但我们可以改写。已知
$$
x_t = x_0 + \sigma_t \epsilon
$$

$$
x_{t+1} = x_0 + \sigma_{t + 1} \epsilon
$$

根据高斯噪声的叠加性，从 $x_t$ 到 $x_{t + 1}$ 仅需加上方差为 $\sigma_{t+1}^2 - \sigma_{t}^2$ 的零均值高斯噪声。那么
$$
x_{t+1} = x_t + \sqrt{\sigma_{t+1}^2 - \sigma_{t}^2} \epsilon
$$
现在我们把离散时间连续化。假设 $t+1$ 和 $t$ 相差的时间为 $\Delta t$，那么
$$
x_{t + \Delta t} - x_t = \sqrt{\sigma_{t +\Delta t} ^ 2 - \sigma_t^2} \epsilon
$$
根据微分的定义 $\frac{df(t)}{dt}=lim_{\Delta t\rightarrow 0}\frac{f(t+\Delta t) - f(t)}{\Delta t}$，最终可知：
$$
dx = \sqrt{\frac{d[\sigma(t)^2]}{dt}} dw
$$
对照前面的微分方程形式，我们可以知道，$f(x,t)=0$，$g(t) = \sqrt{\frac{d[\sigma(t)^2]}{dt}}$。

此外，由于随着时间 $t$ 越来越大，NCSN 的方差是越来越大的；而 DDPM 中，其方差则永远限制在 1 以内，并无限接近 1。所以 NCSN 也被称为 Variance Exploding (VE) SDE，而 DDPM 被称为 Variance Preserving (VP) SDE。

### 微分方程的求解

本文第一节也说到，Euler Method 求解微分方程的数值解容易造成较大的累积误差。所以论文中也提出了一个新的求解方法。具体来说也很简单，每次使用 Euler Method 更新一步后，我们加入纠偏机制，重新修正采样的轨迹。这个纠偏机制就是 NCSN 中使用 Langevin Dynamics 利用 模型预测的 score 去更新 $x$，使得偏离正轨的 $x$ 回道正轨。所以这个方法叫做 Predictor-Corrector 方法（简称 PC）。

具体来说，在每一个时间步（这里我们使用 $\Delta t = 1$），先用 SDE 的方法更新一步。其中 $s_\theta(x_i, \sigma_i)$ 是预测的 score：
$$
x_{i-1} = x_i + (\sigma_i^2 - \sigma_{i-1}^2) s_\theta(x_i, \sigma_i) + \sqrt{\sigma_i^2 - \sigma_{i-1}^2} z
$$
再用 Langevin Dynamics 使用 score 更新 $n$ 步：
$$
x_{i-1} = x_i + \epsilon_i  s_\theta(x_i, \sigma_i) + \sqrt{\epsilon_i} z
$$
具体算法可见[原论文](https://arxiv.org/abs/2011.13456) 附录 G 部分。

### 代码实现

代码实现请见[这里](https://github.com/HoiM/diffusion-schedulers-minimal-implementation/tree/master/04-SDE-VE-PC)。我继续使用了 [NCSN 部分](https://github.com/HoiM/diffusion-schedulers-minimal-implementation/tree/master/03-NCSN)训练出来的模型，因为它正好是训练出来用于预测 score 的。由于我训练的 NCSN 使用了 10 个时间步，所以 Predictor 部分我更新了 10 步；而 Corrector 部分，每次 Predict 后均更新 100 步，你也可以试试别的步数，看看对效果的影响。

### 参考文献

[Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)

[Reverse-time Diffusion Equation Models](https://www.sciencedirect.com/science/article/pii/0304414982900515)