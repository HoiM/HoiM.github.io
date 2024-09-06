# NCSN

NCSN 由 [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600) 这篇论文提出。看题目其实就知道，这种生成算法利用了数据分布的梯度。

我们一般认为，给定一个数据（比如 RGB 图片数据）的分布空间，其符合某种概率分布，那么它的概率密度函数中，真实数据（比如真实图片）或接近真实数据的这块区域的概率密度就高；而数据越不真实（比如随机噪声），其对应的区域概率密度越小。

对于一个函数，在任意位置上，梯度方向是使函数值上升最快的方向。所以，如果我们能得到想要生成的数据分布的概率密度函数的梯度，从随机噪声点开始走，沿着梯度方向一步一步更新，最终我们就可以到达一个概率密度高的点，这个点也就可以作为生成的真实数据了。

### 基本概念

##### Langevin Dynamics

这个概念原本是个物理概念，但我们先不管，把它放在数据分布的语境下：
$$
x_{t+1}=x_t + \tau \nabla_x log(x_t) + \sqrt{2\tau} z
$$
其中 $z$ 是标准高斯噪声。观察这个式子，是不是和前面去噪的公式有点神似。结合前面关于数据的梯度的解释，其实这就是个从低概率密度区域（例如不真实图片的区域）走向高概率密度区域（例如真实图片的区域）的一个过程。在 DDPM 或 DDIM 中这叫去噪过程（这里从 $x_0$ 走到 $x_T$ 是去噪，和 DDPM 等算法用的下标顺序相反）。而 $z$ 这一项，不过是给这个走动的过程加入一些随机性。

##### Stein's Score Function

NCSN 全称为 Noise Conditional Score Network，是用来预测 score 的。什么是 score 呢？这里的 score 是 Stein's Score Function 的简称，它定义为：
$$
s_\theta(x) = \nabla_x logp_\theta(x)
$$
等号左边就是这个 score 的函数，右边就是它的精确定义。其实就是概率密度函数对数据本身求梯度。前面已经说过了梯度的作用，那么其实我们如果有了现成的 score 函数，对于每一个可能的数据 $x$ 位置点，我们都能拿到这个梯度，我们就能根据 Langevin Dynamics 的公式生成真实数据（注意这里的梯度是对 $x$ 求梯度，不是对神经网络的参数；前者是 $x$ 在变，后者 $x$ 不变，网络参数变，更像是极大似然的方法）。

### SMLD: Score Matching Langevin Dynamics

现在问题来到，我们如何拿到这个 score 的函数，让它能对于任意 $x$ 都能给出一个梯度呢？很自然地想到神经网络。我们训练一个神经网络充当 score 函数就行了。这就是 Score Matching 的含义。但有监督训练通常需要“一个输入、一个期望的预测目标”这样的成对数据，我们怎么拿到“期望的预测目标”呢？

理论方法是存在的，但一般比较麻烦、实现困难（感兴趣可以读原论文以及原论文作者的其它论文）。这里作者提出了一个简单的方法：造数据；论文中这被称为 Denoising Score Matching。具体来说，在拿到一些真实图片 $x'$时，我们可以对真实图片加噪，得到 $x$：
$$
x = x' + \sigma z
$$
其中，$z$ 是一个标准高斯噪声，$\sigma$ 决定了高斯噪声的标准差。这时，$q(x)$ 可以用 $q(x|x')$ 替代。对 0 均值、方差为 $\sigma^2$ 的高斯分布加上一个 $x'$，可以得到其分布的具体形式就是一个均值为 $x'$、方差为 $\sigma^2$ 的高斯分布。对它的概率密度函数求导（具体过程可参考[这篇论文](https://arxiv.org/abs/2403.18103)第 36 页）可得 score 是：
$$
\nabla_x logq(x|x') = -\frac{z}{\sigma^2}
$$
这样，神经网络训练所需要的输入和目标输出就都有了，就可以训练网络了。

### 模型训练

我们用 $s_\theta$ 代表要训练的神经网络，其中 $\theta$ 是它的参数。需要注意的是，我们可能从数据分布的空间的任意一个起始点出发，走向高概率密度区域。那么上面提到的 $x=x' + \sigma z$ 公式中，$x$ 需要足够多样。为此我们需要多个不同的 $\sigma$，其值越大，得到的加噪数据就越远离真实数据。通过不同的 $\sigma$、足够多次的采样，训练的模型才能学到数据空间里不同位置的 score 是什么。 

另外，针对不同的 $\sigma$，其损失还需要不同的权重。也就是说，对于 $\sigma \in \{\sigma_i\}_{i=1}^L$，有：
$$
L(\theta, \{\sigma_i\}_{i=1}^L)={1\over L} \sum_{i=1}^L \lambda(\sigma_i)l(\theta, \sigma_i)
$$
这时由于当 $\sigma$ 不同时，其真实 score 的大小会有差别（观察上一节最后的公式）。为了平衡不同 loss 的幅值，我们设置 $\lambda(\sigma_i) = \sigma_i^2$。

至于 $\sigma_i$ 的取值，我们遵循从大到小的顺序，且有 ${\sigma_0 \over \sigma_1}=…={\sigma_{L-1} \over \sigma_L} > 1$。

### 采样：annealed Langevin Dynamics

最终，NCSN 提出了一种叫作 annealed Langevin Dynamics 的采样算法来生成样本。具体步骤如下图：

![annealed-langevin-dynamics.png](https://github.com/HoiM/HoiM.github.io/blob/main/assets/annealed-langevin-dynamics.png?raw=true)

其中 $x_0$ 是我们随机采样的一个起点。然后 $\sigma$ 从大到小（因为一开始数据距离真实数据远，而大的 $\sigma$ 对应距离真实数据远的样本）进行采样。针对每一个 $\sigma$ 我们迭代 $T$ 次，这里就使用了 Langevin Dynamics 的公式。注意到算法中还有一个 $\epsilon$，这个数值和“梯度更新的步长”相关。

### 代码实现

在[这里](https://github.com/HoiM/diffusion-schedulers-minimal-implementation/tree/master/03-NCSN)我实现了简易的 NCSN 算法，数据集仍然用 MNIST，网络结构和前面 DDPM 一致。我发现训练 NCSN 需要的迭代次数显著更多，这里我用了 750 个 epoch 训练，效果才差不多。

### 参考文献

[Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600)

[Tutorial on Diffusion Models for Imaging and Vision](https://arxiv.org/abs/2403.18103)