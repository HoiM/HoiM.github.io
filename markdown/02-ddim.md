# DDIM

DDPM 问世后，还存在一个问题，就是采样需要的时间步太长了。在论文中，作者提出使用 1000 步去采样生成。在[我的 DDPM 实现](https://github.com/HoiM/diffusion-schedulers-minimal-implementation/tree/master/01-DDPM)中，用 MNIST 训练生成手写数字，也需要 250 步。这一特性严重增加了模型推理时间。而 DDIM 则旨在解决这一问题：我们不用推理完整的 1000 步或 250 步，我们其实可以跳跃地采样。

### 核心原理

在 DDPM 论文中，加噪去噪过程被设计成了一个马尔可夫过程，这是一个一环接一环的过程，每个时间步缺一不可。但是 DDIM 发现，其实这个过程可以不是马尔可夫过程。

论文中提出的一个核心结论是，对于这样一组分布：
$$
q_{\sigma}(x_{1:T}|x_0) := q_{\sigma}(x_T|x_0)\prod _{t=2} ^{T} q_{\sigma}(x_{t-1}|x_t, x_0)
$$
其中，$q_{\sigma}(x_T|x_0)$ 服从 $N(\sqrt{\bar\alpha_T} x_0, (1-\bar\alpha_T)I)$ 这样的高斯分布（这也来自 DDPM 中的设计）。存在这样的去噪过程，$q_{\sigma}(x_{t-1}|x_t, x_0) $ 服从如下高斯分布：
$$
N(\sqrt{\bar\alpha_{t-1}}x_0 + \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2} \frac{x_t- \sqrt{\bar{\alpha} _t} x_0}{\sqrt{1-\bar{\alpha} _t}}, \sigma_tI)
$$
关于这里的原理推导，可参考 [DDIM 论文](https://arxiv.org/abs/2010.02502) 附录 B 部分。但我发现[这里](https://zhuanlan.zhihu.com/p/666552214)的 4.1 部分，还有一个非常通俗易懂的推导，更好理解。（注意：DDIM 论文中用 $\alpha_t$ 全面替代了 DDPM 论文中的 $\bar\alpha_t$，这里我还按照 DDPM 论文中的写法，写成 $\bar\alpha_t$，以免混淆）

有了上面的分布，我们就能通过 $x_t$ 和 $x_0$ 进行去噪得到 $x_{t-1}$ 了，但 $x_0$ 从哪来呢？回想一下这个公式：
$$
x_t = \sqrt{\bar{\alpha_t}} x_0 + \sqrt{1 - \bar{\alpha_t}} \epsilon
$$
我们其实可以在 $x_t$、$x_0$、$\epsilon$ 三者之间任意进行转换：
$$
x_0 = \frac {x_t - \sqrt{1 - \bar{\alpha_t}} \epsilon} {\sqrt{\bar{\alpha_t}}}
$$

$$
\epsilon = \frac{x_t -  {\sqrt{\bar{\alpha_t}}} x_0} {\sqrt{1 - \bar{\alpha_t}}}
$$

把这两个式子带入上面 $q_{\sigma}(x_{t-1}|x_t, x_0) $ 服从的高斯分布，就可以得到：
$$
x_{t-1} = \sqrt{\bar\alpha_{t - 1}} (\frac{x_t - \sqrt{1 - \bar\alpha_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar\alpha_t}}) + \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\epsilon_\theta(x_t, t) + \sigma_t\epsilon_t
$$
这就是 DDIM 去噪的核心公式了，其中 $\epsilon_\theta$ 是模型预测出来的噪声，$\epsilon_t$ 是标准高斯噪声，用来在采样过程中加入随机性 。但整个公式和跳步采样有什么关系？观察上面的推导过程，其实全程并为用到 DDPM 里马尔可夫公式，即 $x_{t} = \sqrt{1-\beta_{t-1}} x_{t-1} + \sqrt{\beta_{t-1}} \epsilon$ 这个公式始终没用到。那上面去噪公式里 $t$ 和 $t-1$ 其实并不一定是相邻的两步（比如第 249 步和第 248 步），它仅仅代表一种时间步的先后关系（比如第 249 步和第 229 步）。这样就实现了跳跃式的采样过程。

还有一个问题，$\sigma_t$ 怎么得到？如果我们拿上面的公式，合并所有 $\epsilon_\theta$ 的项，与 DDPM 的去噪公式里 $\epsilon$ 的系数对比，就可以求解出来：
$$
\sigma_t = \sqrt{\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}} \sqrt{1-\frac{\bar\alpha_t}{\bar\alpha_{t-1}}}
$$
当然这只是一种 $\sigma_t$ 的设计。如果按照这个式子，前向过程就成了马尔可夫过程，整个扩散模型就变成了 DDPM（如果你不跳步采样的话）。实际情况下，我们一般会 $\sigma_t \epsilon_t$ 这一项使用：
$$
\sigma_t = \eta \sqrt{\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}} \sqrt{1-\frac{\bar\alpha_t}{\bar\alpha_{t-1}}}
$$
其中 $\eta$ 在 0 到 1 之间取值。当 $\eta$ 为 0 时，整个过程就成了确定性的采样，没有随机性。

### 代码实现

我在[这里](https://github.com/HoiM/diffusion-schedulers-minimal-implementation/tree/master/02-DDIM)实现了 DDIM 的采样代码，这里使用了我前面 DDPM 训练出来的模型和权重，采样 MNIST 样式的手写数字。这里我使用了 50 步采样，相比于我训练的 250 的 DDPM，推理时间可减少到五分之一。

### 参考文献

[Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)





