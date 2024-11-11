# HW1

## 1 PAC Bound

​	Using $C$ to denote the range of the loss function, we have:
$$
C = C_2 - C_1 \tag{1}
$$
​	Then, accordign to **Hoeffding's inequality**, for $\forall h \in \mathcal{H}$:
$$
P\left(|L_s(h) - L_D(h)| \geq \epsilon\right) \leq 2 e^{-\frac{2m\epsilon^2}{C^2}} \tag{2}
$$
​	To ensure that this holds for $\forall h \in \mathcal{H}$, we apply the **union boud**:
$$
P\left(\exists h \in \mathcal{H}: |L_s(h) - L_D(h)| \geq \epsilon\right) \leq 2|\mathcal{H}| e^{-\frac{2m\epsilon^2}{C^2}} 
\tag{3}
$$
​	As we want probability to be at MOST $\delta$, we have:
$$
2|\mathcal{H}| e^{-\frac{2m\epsilon^2}{C^2}} \leq \delta
\tag{4}
$$
​	We can solve for $\epsilon$ then:
$$
\begin{gather*}
e^{-\frac{2m\epsilon^2}{C^2}} \leq \frac{\delta}{2|\mathcal{H}|}\\
-\frac{2m\epsilon^2}{C^2} \leq \ln{\frac{\delta}{2|\mathcal{H}|}}\\
\epsilon^2 \geq \frac{C^2}{2m} \ln{\frac{2|\mathcal{H}|}{\delta}}\\
\epsilon \geq \frac{C}{\sqrt{2m}}\sqrt{\ln{\frac{2|\mathcal{H}|}{\delta}}} = \frac{C_2-C_1}{\sqrt{2m}}\sqrt{\ln{\frac{2|\mathcal{H}|}{\delta}}} \tag{5}
\end{gather*}
$$
​	Thus, for any finite hypothesis space of $\mathcal{H}$ and for any learned function $\tilde{h} = \arg\min_{h \in \mathcal{H}} L_s(h)$ with probability $1-\delta$, the generalization bound for the true risk $L_D(\tilde{h})$ is:
$$
L_D(\tilde{h}) \leq L_S(\tilde{h}) + \frac{C_2-C_1}{\sqrt{2m}}\sqrt{\ln{\frac{2|\mathcal{H}|}{\delta}}} \tag{6}
$$

## 2 Laplacian Matrix

**2.1 Prove Laplacian matrix $L$ is positive-senidefinite.**

​	To prove Laplacian matrix $L$, we need to prove that: 
$$
\forall\boldsymbol{x} \in \mathbb{R}^n,\ \boldsymbol{x}^T L \boldsymbol{x} \geq 0
\tag{1.1}
$$
​	According to the definition, we can expand $\boldsymbol{x}^T L \boldsymbol{x}$:
$$
\boldsymbol{x}^T L \boldsymbol{x} = \boldsymbol{x}^T (D-A) \boldsymbol{x} = \boldsymbol{x}^T D \boldsymbol{x} - \boldsymbol{x}^T A \boldsymbol{x}
\tag{1.2}
$$
​	For those 2 terms in equation (1.2):
$$
\boldsymbol{x}^T D \boldsymbol{x} = \sum_{i=1}^n \sum_{i=1}^n D_{ii} \boldsymbol{x}_i^2 = \sum_{i=1}^n \text{deg}(i) \cdot \boldsymbol{x}_i^2
\tag{1.3}
$$
in which $\text{deg}(i)$ denotes the degree of i-th node.
$$
\boldsymbol{x}^T A \boldsymbol{x} = \sum_{i=1}^n\sum_{j=1}^n A_{ij} \boldsymbol{x}_i \boldsymbol{x}_j
\tag{1.4}
$$
​	Using $\mathcal{N}_i$ to denote the neighborhood of i-th node, we can rewrite equation (1.3) & (1.4):
$$
\begin{gather*}
\sum_{i=1}^n \text{deg}(i) \cdot \boldsymbol{x}_i^2 = \sum_{i=1}^n\sum_{j \in \mathcal{N}_i} \boldsymbol{x}_i^2
\tag{1.5}\\
\sum_{i=1}^n\sum_{j=1}^n A_{ij} \boldsymbol{x}_i \boldsymbol{x}_j
 = \sum_{i=1}^n\sum_{j \in \mathcal{N}_i} \boldsymbol{x}_i \boldsymbol{x}_j
 \tag{1.6}
\end{gather*}
$$
​	By subsituting equation (1.5) & (1.6) into the equation (1.2), we can obtain:
$$
\boldsymbol{x}^T L \boldsymbol{x} = \sum_{i=1}^n\sum_{j \in \mathcal{N}_i} \boldsymbol{x}_i^2 - \sum_{i=1}^n\sum_{j \in \mathcal{N}_i} \boldsymbol{x}_i \boldsymbol{x}_j
\tag{1.7}
$$
​	Then, consider equation (1.7) in edge-wise form, it can be rewritten as:
$$
\begin{align*}
\sum_{i=1}^n\sum_{j \in \mathcal{N}_i} \boldsymbol{x}_i^2 - \sum_{i=1}^n\sum_{j \in \mathcal{N}_i} \boldsymbol{x}_i \boldsymbol{x}_j &= \sum_{(i,j)\in E}
\left[\left(
\boldsymbol{x}_i^2 +\boldsymbol{x}_j^2 
\right) - 2\boldsymbol{x}_i\boldsymbol{x}_j
\right] \\
&= \sum_{(i,j)\in E} \left(\boldsymbol{x}_i - \boldsymbol{x}_j\right)^2 \geq 0
\tag{1.8}
\end{align*}
$$
in which $E$ denotes the edge set of the given graph, $(i,j),\ i\lt j$ denotes one of the undirected edges.

​	Therefore, $\boldsymbol{x}^T L \boldsymbol{x} \geq 0$ holds for $\forall\boldsymbol{x} \in \mathbb{R}^n$. Hence, Laplacian matrix $L$ is positive-senidefinite.

---

**2.2 Let define normalized Laplacian matrix as $\hat{L}=D^{-1/2}LD^{-1/2}$, derive the upper/lower bound of the eigenvalues of $\hat{L}$.**

​	According to the definition, we have:
$$
\begin{align*}
\hat{L}&= D^{-1/2}LD^{-1/2} = D^{-1/2}(D-A)D^{-1/2}\\
&=I- D^{-1/2}AD^{-1/2}
\tag{2.1}
\end{align*}
$$
​	Assuming that:
$$
A = \left[\begin{array}{1}
a_{11} & a_{12} & ... & a_{1n} \\
a_{21} & a_{22} & ... & a_{2n} \\
\vdots & \vdots &     & \vdots \\
a_{n1} & a_{n2} & ... & a_{nn}
\end{array}\right],\ D = \left[\begin{array}{1}
d_1 &     &     &  \\
    & d_2 &     &  \\
    &     & ... & \\
    &     &     & d_n
\end{array}\right]
\tag{2.2}
$$
​	Then:
$$
\begin{align*}
D^{-1/2} &= \left[\begin{array}{1}
\frac{1}{\sqrt{d_1}} &     &     &  \\
    & \frac{1}{\sqrt{d_2}}  &     &  \\
    &     & ... & \\
    &     &     & \frac{1}{\sqrt{d_n}} 
\end{array}\right]\\
D^{-1/2}AD^{-1/2} &= \left[\begin{array}{1}
\frac{a_{11}}{\sqrt{d_1d_1}} & \frac{a_{12}}{\sqrt{d_1d_2}} & ... & \frac{a_{1n}}{\sqrt{d_1d_n}} \\
\frac{a_{21}}{\sqrt{d_2d_1}} & \frac{a_{22}}{\sqrt{d_2d_2}} & ... & \frac{a_{2n}}{\sqrt{d_2d_n}} \\
\vdots & \vdots &  & \vdots \\
\frac{a_{n1}}{\sqrt{d_nd_1}} & \frac{a_{n2}}{\sqrt{d_nd_2}} & ... & \frac{a_{nn}}{\sqrt{d_nd_n}} \\
\end{array}\right]
\tag{2.3}
\end{align*}
$$
​	For arbitrary vector $\boldsymbol{x} \in \mathbb{R}^n$:
$$
\begin{align*}
\boldsymbol{x}^T(D^{-1/2}AD^{-1/2})\boldsymbol{x} &= [x_1, x_2, ..., x_n]
\left[\begin{array}{1}
\frac{a_{11}}{\sqrt{d_1d_1}} & \frac{a_{12}}{\sqrt{d_1d_2}} & ... & \frac{a_{1n}}{\sqrt{d_1d_n}} \\
\frac{a_{21}}{\sqrt{d_2d_1}} & \frac{a_{22}}{\sqrt{d_2d_2}} & ... & \frac{a_{2n}}{\sqrt{d_2d_n}} \\
\vdots & \vdots &  & \vdots \\
\frac{a_{n1}}{\sqrt{d_nd_1}} & \frac{a_{n2}}{\sqrt{d_nd_2}} & ... & \frac{a_{nn}}{\sqrt{d_nd_n}} \\
\end{array}\right]
\left[\begin{array}{1}
x_1\\x_2\\ \vdots \\ x_n
\end{array}\right] \\
&= \left[ \sum_{i=1}^n \frac{x_ia_{i1}}{\sqrt{d_id_1}}, \sum_{i=1}^n \frac{x_ia_{i2}}{\sqrt{d_id_2}}, ... , \sum_{i=1}^n \frac{x_ia_{in}}{\sqrt{d_id_n}}
\right] \left[\begin{array}{1}
x_1\\x_2\\ \vdots \\ x_n
\end{array}\right] \\
&= \sum_{i=1}^n \frac{x_ix_1a_{i1}}{\sqrt{d_id_1}} + \sum_{i=1}^n \frac{x_ix_2a_{i1}}{\sqrt{d_id_2}} + ... + \sum_{i=1}^n \frac{x_ix_na_{i1}}{\sqrt{d_id_n}} \\
&= \sum_{i=1}^n\sum_{j=1}^n \frac{x_ix_ja_{ij}}{\sqrt{d_id_j}} = \sum_{(i,j)\in E} \frac{2x_ix_ja_{ij}}{\sqrt{d_id_j}}
\tag{2.4}
\end{align*}
$$
​	Thus, the quadratic form of $\hat{L}$ is:
$$
\begin{align*}
\boldsymbol{x}^T\hat{L}\boldsymbol{x} &= \boldsymbol{x}^T(I- D^{-1/2}AD^{-1/2})\boldsymbol{x} \tag{according\ to\ 2.1}
\\
&= \boldsymbol{x}^T\boldsymbol{x}-\boldsymbol{x}^T(D^{-1/2}AD^{-1/2})\boldsymbol{x}\\
&=\sum_{(i,j)\in E}\left(\frac{x_i^2}{d_i}+\frac{x_j^2}{d_j}\right) - \sum_{(i,j)\in E} \frac{2x_ix_ja_{ij}}{\sqrt{d_id_j}} \tag{according\ to\ 2.4}
\\
&=\sum_{(i,j)\in E}\left(\frac{x_i}{\sqrt{d_i}}-\frac{x_j}{\sqrt{d_j}}\right)^2
\geq 0
\tag{2.5}
\end{align*}
$$

---

**Lemma**:

​	For $n \times n$ symmertric matrix $A$ & arbitrary vector $\boldsymbol{x} \in\mathbb{R}^n$, we can define **Rayleigh quotient**:
$$
R(A,\boldsymbol{x}) = \frac{\boldsymbol{x}^TA\boldsymbol{x}}{\boldsymbol{x}^T\boldsymbol{x}}
$$
​	And if the eigenvalues of $A$ are $\lambda_1 \leq \lambda_2 \leq ... \leq \lambda_n$, then $\lambda_1 \leq R(A,\boldsymbol{x}) \leq \lambda_n$.

---

​	We can have the <u>upper bound</u> of $R(D^{-1/2}AD^{-1/2},\ \boldsymbol{x})$ then:
$$
\begin{gather*}
\boldsymbol{x}^T\hat{L}\boldsymbol{x}=\boldsymbol{x}^T\boldsymbol{x}-\boldsymbol{x}^T(D^{-1/2}AD^{-1/2})\boldsymbol{x} \geq 0 \tag{according\ to\ 2.5}
\\
\Rightarrow \frac{\boldsymbol{x}^T(D^{-1/2}AD^{-1/2})\boldsymbol{x}}{\boldsymbol{x}^T\boldsymbol{x}} = R(D^{-1/2}AD^{-1/2},\ \boldsymbol{x})\leq 1\tag{2.6}
\end{gather*}
$$
​	Simmilarly, we can also prove the <u>lower bound</u> of $R(D^{-1/2}AD^{-1/2},\ \boldsymbol{x})$:
$$
\begin{align*}
\boldsymbol{x}^T(I+ D^{-1/2}AD^{-1/2})\boldsymbol{x}&=\boldsymbol{x}^T\boldsymbol{x}+\boldsymbol{x}^T(D^{-1/2}AD^{-1/2})\boldsymbol{x}\\
&= \sum_{(i,j)\in E}\left(\frac{x_i^2}{d_i}+\frac{x_j^2}{d_j}\right) +\sum_{(i,j)\in E} \frac{2x_ix_ja_{ij}}{\sqrt{d_id_j}} \geq 0\\
& \Rightarrow\frac{\boldsymbol{x}^T(D^{-1/2}AD^{-1/2})\boldsymbol{x}}{\boldsymbol{x}^T\boldsymbol{x}} =R(D^{-1/2}AD^{-1/2},\ \boldsymbol{x})\geq -1 \tag{2.7}
\end{align*}
$$
​	$\hat{L}$'s Rayleigh quotient:
$$
R(\hat{L}, \boldsymbol{x}) = \frac{\boldsymbol{x}^T\hat{L}\boldsymbol{x}}{\boldsymbol{x}^T\boldsymbol{x}} = \frac{\boldsymbol{x}^T(I - D^{-1/2}AD^{-1/2})\boldsymbol{x}}{\boldsymbol{x}^T\boldsymbol{x}}=1-R(D^{-1/2}AD^{-1/2},\ \boldsymbol{x}) \tag{2.8}
$$
​	According to (2.6) & (2.7), we have $R(D^{-1/2}AD^{-1/2},\ \boldsymbol{x}) \in [-1,1]$. Hence, the range of $R(\hat{L}, \boldsymbol{x})$ is:
$$
0 \leq R(\hat{L}, \boldsymbol{x}) \leq 2 \tag{2.9}
$$
​	

​	Applying the **Lemma**, we can prove that the eigenvalues of $\hat{L}$ also in range $[0,2]$. 

​	Thus, the <u>lower bound</u> for the eigenvalues of $\hat{L}$ is 0 and the <u>upper bound</u> is 2.