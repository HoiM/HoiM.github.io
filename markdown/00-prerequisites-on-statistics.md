# 统计学基础

### 贝叶斯公式

贝叶斯公式：
$$
P(H|E)={P(E|H)\times P(H)\over P(E)}
$$
其中 H 代表 hypothesis（假设），E 代表 evidence（依据）。在我们观察到的证据下，我们希望得到某个假设成立的可能性，就可以用以上公式。

举例：我咳嗽（evidence）了，我想知道我被 covid 感染（hypothesis）的概率有多大。那么，如果很多疾病都有咳嗽的症状（$P(E)$ 比较大），那么我感染 covid 的可能就小（分母大，所以整个公式的值就小）；如果全国人民都在感染 covid 的高峰期（$P(H)$ 比较大）那我感染的可能也更大（分子大，所以值就大）；如果 covid 一个典型症状就是咳嗽（$P(E|H)$ 比较大），那我感染 covid 的可能性也会大（分子大，所以值就大）。

### Evidence Lower Bound (ELBO)

在 VAE 等生成模型的语境下，evidence 一般是我们观测到的数据。比如我们想要生成一些图片，那我们搜集了图片数据（一般用 $x$ 表示）。同时我们还需要隐变量（latent，一般用 $z$ 表示）。通常，我们需要一个模型，能够将 $x$ 转换到 $z$，也就是 $p(z|x)$。但这个比较困难，所以，一种办法是通过一个分布 $q(z)$，去近似 $p(z|x)$。

由于似然函数 $p(x)$ 可以写成这样（这里都加了 log）：
$$
log{p(x)}=log\int{p(x,z)}dz
$$
然后我们引入 $q(z)$:
$$
log{p(x)}=log\int{p(x,z)\over q(z)} q(z)dz
$$
根据 Jensen's Inequality，对于一个凹函数 $\phi $，有：（这个画个图很好理解）
$$
\phi(E(X)) \ge E(\phi (X))
$$
把 $\phi $ 换成 log 函数，把在 $q(z)$ 下的积分（marginalize）看成求期望，套用 Jensen's Inequality，就能得到：
$$
log{p(x)}=logE_{q(z)}[{p(x,z)\over q(z)}] \ge E_{q(z)}[log{p(x,z)\over q(z)}]=E_{q(z)}[log{p(x|z)}]
$$
所以：
$$
log{p(x)} \ge E_{q(z)}[log{p(x,z)\over q(z)}]
$$
或者：
$$
log{p(x)} \ge E_{q(z)}[log{p(x|z)}]
$$


式子右边就是 ELBO，对它最大化，就间接对对数似然函数进行了最大化。同时也能求出来 $q(z)$ 来近似 $p(z|x)$。

### VAE

VAE 的训练可以看作是一种最大似然方法。前面说了，最大化 ELBO 就间接最大化了 $logp(x)$，这就是一种最大似然。而最大化 $logp(x|z)$，在实际实现上，就是 decoder 拿到 $z$ 后，生成图片，并和真实的图片做一个 loss。当然训练中还有诸如 reparameterization 等步骤（让 encoder 预测均值和方差，而不是直接预测 $z$），这里不重点讲解了。

### 参考文献

除了这种推导和理解外，以下两篇论文也都给出了很好的解释：

[Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970)

[Tutorial on Diffusion Models for Imaging and Vision](https://arxiv.org/abs/2403.18103)