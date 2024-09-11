# 机器学习基础

> 如何缓解 Overfitting？
>
> 加数据，加数据就好；然后搞点数据增强

## 基础分类

1. 监督学习：数据中既有 training data，也有 label

    经典：Regression（Y-Label 是连续的实值函数）/ Claasification（Y-Label 是离散的）

2. 无监督学习：数据只有 training data

    经典：Clustering（聚类）/ Representation Learning（表征学习）

    > 自监督学习：包括 生成模型 + 对比模型
    >
    > - 生成（压缩，再还原）；对比（数据增强，对比表征相似度）
    >
    > - 让算法自行从 training data 中生成 label，使得 “无监督学习->监督学习”
    >
    > - 通过 数据 + 人工先验知识 来构造标签


3. 强化学习：对于一系列 actions 进行奖励（反馈）

    在与环境的交互过程中，逐步获得更优解法

## 基础知识储备

- Domain Set $X$：样本的定义域 => 特征可以取到哪些空间

- Label Set $Y$

- Training Data $S = ((x_1,y_1),...(x_m,y_m))$：从训练数据分布 $D = X \times Y$ 中采样得到的有限序列

- Learner：学习得到函数 $h$ 使得 $h(X) \rightarrow Y$，其超参数 $H$ 能根据数据自适应的进行调整

---

- Loss Function $l$：用于对误差进行评估

- True Risk：在测试集中的表现

    ML 的目标是 True Risk 下降（而非 Loss 下降）

    训练集误差 $\neq$ 测试集误差，训练集误差 $=0$ 不代表在测试集中表现良好

---

### PAC 学习理论

对于任何有限的 hypothesis space $H$：

- 在训练集中得到的分类器 $\hat{h}$

    > 通过 minimize 训练集 Loss 得到（但有时候并不能很好的 minimize）：
    >
    > - loss 不是凸函数时，只能收敛到局部最小
    >
    > - 有时候目标不连续，甚至不可导

    所以：数据 & 优化器 都很重要

- 测试集上的最优分类器 $h^*$（也在 $H$ 规定的空间内）

有 $1-\delta$ 的概率满足：

$$
L_D(\hat{h}) \leq L_D(h^*) + \sqrt{\frac{2}{m}\log{\frac{2|H|}{\delta}}}\text{(Theorem\ 1)}
$$

其中，$m$ 为训练集大小、$H$ 为假设空间大小，这两个参数共同决定了两者间的差距

> Overfitting 是 训练集太小 + 假设空间太大 导致的

---

#### Proof of Theorem 1

##### 1 建立 $L_D(h), L_S(h)$ 之间的联系
 
1. 中心极限定律（Hoeffding's Inequality）： 
    
    有 $m$ 个服从独立分布的、从同一分布中采样的变量 $\theta_1,...\theta_m$ ，其期望 $E|\theta_i|$ 均为 $\mu$，且 $\theta \in [a,b]$ 有界

    对于任意 $\epsilon > 0$，满足：

    $$
    P[|\frac{1}{m}\sum_{i=1}^m(\theta_i-\mu)|>\epsilon] \leq 2 \exp{(-\frac{2m\epsilon^2}{(b-a)^2})}
    $$

    > 采样均值与原始分布均值之间的关系：采样越多、差距越小

    而 $L_D(h)$ 是整个样本空间里的采样，而 $L_S(h)$ 是样本空间分布的有限采样（设就是联系！），有

    $$
    P(|L_S(h) - L_D(h)|) \gt \epsilon \leq 2 \exp{(-2m\epsilon^2)}
    $$

    - 因为假设 Loss $\in [0,1]$，这里 $(b-a)^2$ 就消掉了

    - 所以，当 $m$ 趋近于无穷时，我们就可以用 $L_S(h)$ 去估计 $L_D(h)$

2. 上式对于某一个特定的 $h()$ 成立，将这个事件记为 $A_i$。进一步的，由于：

    $$
    P(A_1 \cup ... \cup A_n) \leq \sum_i{i=1}^nP(A_i)
    $$

    有：

    $$
    P(\{\exist h \in H, |L_S(h) - L_D(h)|) \gt \epsilon \}) \leq 2|H|\exp{(-2m\epsilon^2)}
    $$

    ---

    记 $\delta = 2|H|exp(-2m\epsilon^2)$，可改写为：

    $$
    P(\{\exist h \in H, |L_S(h) - L_D(h)|) \leq \epsilon \}) \leq 1-\delta
    $$

##### 2 建立 $L_D(\hat{h}), L_D(h^*)$ 之间的联系

由于 $|L_S(h) - L_D(h)| \leq \epsilon$，有：

$$
\begin{align*}
    L_D(\hat{h}) &\leq L_S(\hat{h}) + \epsilon \\
    &\leq L_S(h^*) + \epsilon\\
    &\leq L_D(h^*) + 2\epsilon
\end{align*}
$$

### Bias-Complexity 分解

- $e_a = L_D(h^*)$ 为 inductive bias 

    不可学习的内置偏差

- $e_c = L_D(\hat{h})-L_D(h^*)$ 为 estimation error

    由学习导致的误差，可以由 $(m,H)$ 进行约束


> - Underfitting（$H$ 选的不好、太小了）
>
> - Overfitting（$H$ 选的大，但是 $m$ 太小学歪了）

