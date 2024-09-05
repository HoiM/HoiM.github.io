# DDPM

DDPM 设计了一个马尔可夫过程。在加噪过程中，每个时间步上对当前的图片进行一定程度的退化；而去噪过程则根据贝叶斯公式得到了每一时间步的高斯分布的均值和方差。

### 前向过程（加噪）

前向马尔可夫过程（从真实图片 $x_0$ 到高斯噪声 $x_T$）如下：
$$
x_{t} = \sqrt{1-\beta_{t-1}} x_{t-1} + \sqrt{\beta_{t-1}} \epsilon
$$
其中，$\epsilon$ 是标准高斯噪声（均值为 0 方差为 1），$\beta_t$ 随着 $t$ 的增大而增大（所以噪声的比重越来越大）。通常我们也有 $\alpha_t=1-\beta_t$，所以上面公式也变成：
$$
x_{t} = \sqrt{\alpha_{t-1}} x_{t-1} + \sqrt{1-\alpha_{t-1}} \epsilon
$$
根据高斯分布的叠加性，层层带入，也可以得到 $x_t$ 关于 $x_0$ 的公式（其中 $\bar{\alpha_t} = \alpha_1\alpha_2…\alpha_t$）：
$$
x_t = \sqrt{\bar{\alpha_t}} x_0 + \sqrt{1 - \bar{\alpha_t}} \epsilon
$$
此外，根据上面公式，还能得到：
$$
x_0 = \frac {x_t - \sqrt{1 - \bar{\alpha_t}} \epsilon} {\sqrt{\bar{\alpha_t}}}
$$

### 反向过程（去噪）

反向过程可以写成求解 $p(x_{t-1}|x_t)$ 的过程。由于 $x_{t - 1}$ 和 $x_t$ 都是基于 $x_0$ 产生的，我们可以将 $p(x_{t-1}|x_t)$ 写成 $p(x_{t-1}|x_t， x_0)$。根据贝叶斯公式，有（竖线右边都加上 $x_0$）：
$$
p(x_{t-1}|x_t, x_0) = \frac {p(x_{t}|x_{t-1}, x_0)p(x_{t-1}|x_0)} {p(x_t|x_0)}
$$
根据前向过程的几个公式，上式等号右边三个分布我们都是已知的，带入各自的高斯分布，并用前向过程部分最后一个公式替换掉 $x_0$，就能得到  $p(x_{t-1}|x_t)$ 。它也是一个高斯分布：
$$
N(\frac{1}{\sqrt{\alpha_t}} (x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha_t}}} \epsilon), \frac{(1-\alpha_t)(1-\bar\alpha_{t-1})}{1-\bar\alpha_t} )
$$
这里省略了复杂的推理过程，感兴趣可参考[这篇论文](https://arxiv.org/abs/2208.11970)。

### 模型训练

观察  $p(x_{t-1}|x_t)$ 这个高斯分布，我们发现在反向过程中，只有加的噪声 $\epsilon$ 是未知的。所以我们就训练一个神经网络，输入 $x_t$ 和 $t$，让它预测在特定的 $t$ 时刻，噪声是什么。

具体来说，每次迭代，我们随机生成一个标准高斯噪声 $\epsilon$，再挑选一个时间步 $t$，根据前面 $x_0$ 到 $x_t$ 的公式，得到 $x_t$。把 $x_t$ 和 $t$ 送入神经网络让它预测先前的随机噪声：
$$
L=||\epsilon_\theta(x_t, t) - \epsilon||^2
$$
其中 $\epsilon_\theta$ 就代表要训练的神经网络。

严谨的推导也可以通过 VAE 的 Evidence Lower Bound 推出，其中隐变量 $z$ 在这里变成了 $x_1,x_2,…,x_T$ 的联合分布，我们要求的是 $x_0$。感兴趣可参考[这篇论文](https://arxiv.org/abs/2208.11970)。

![ddpm-algo.png](https://github.com/HoiM/HoiM.github.io/assets/ddpm-algo.png)

### 采样

采样根据上图 Algorithm 2 的算法迭代 T 步即可。其中 $\sigma_t$ 即可使用反向过程中高斯分布中的方差，也可使用前向过程中的 $\beta_t$（详见 [DDPM 论文](https://arxiv.org/abs/2006.11239)）。其实就算不加这个方差的扰动，也可以生成出图片。

### 代码实现

我在[这里](https://github.com/HoiM/diffusion-schedulers-minimal-implementation/tree/master/01-DDPM)实现了一个最小化的 DDPM，旨在剔除冗余代码，让代码尽可能易读，最小化地体现实现要素。数据使用了 MNIST 来训练，在使用 250 个时间步、训练 20 个 epoch 的情况下，即可生成不错的效果。

### 参考文献

[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

[Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970)