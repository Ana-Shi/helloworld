---

---



# Softmax



## 0.说在前面

今天来学习Softmax梯度推导及实现！

有关更多内容，请关注微信公众号：guangcity

![](https://github.com/Light-City/images/blob/master/wechat.jpg?raw=true)

## 1.损失函数

> 矩阵乘法

矩阵相乘，矩阵A的一行乘以矩阵B的每一列，不用循环B矩阵乘法公式：

对于下面这个，则不用循环W矩阵，否则通常做法还得循环W矩阵的每一列！

```python
score = np.dot(X[i],W)
```

> 损失函数

具体的描述看代码，有一点需要注意，损失函数Loss也就是cross-entropy！

在实际计算的时候，需要给分子分母同时乘以常熟C，一般C取-maxfj，目的是防止数值爆炸，所产生的导致计算机内存不足，计算不稳定！

```python
def softmax_loss_naive(W, X, y, reg):
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_class = W.shape[1]
  for i in range(num_train):
      # 得到S矩阵每一行
      score = np.dot(X[i],W)
      # 防止数值爆炸，保持稳定性
      score-=max(score)
      # 分子 去指数
      score = np.exp(score)
      # 分母，S矩阵每一行求和
      softmax_sum = np.sum(score)
      # broadcast：向量除以标量
      score /= softmax_sum
      # 得到交叉熵，也就是softmax的loss
      loss -= np.log(score[y[i]])
   # 平均         
   loss/=num_train
   # 加上正则项
   loss+=reg*np.sum(W*W) 
  return loss, dW
```



## 2.梯度推导

> shape查看

X为(D,N)，W为(N,C)

> 梯度求导推论

![](https://latex.codecogs.com/gif.latex?X_iW_j%5ET%3DS_%7Bij%7D)

这里X与Wj转置均是行向量!

记作(2)式：

![](https://latex.codecogs.com/gif.latex?Softmax%28S_%7Bij%7D%29%20%3D%20%5Cfrac%20%7Be%5E%7BS_%7Bij%7D%7D%7D%7B%5Csum_k%7Be%5E%7BS_%7Bik%7D%7D%7D%7D%3Dq_%7Bij%7D)

记作(3)式：

![](https://latex.codecogs.com/gif.latex?L_i%20%3D%20-%5Csum_mp_mlogq_%7Bim%7D)

pm = [0,...1...,0]是一个是一个one hot vector

梯度求导：

利用链式求导法则：记作(4)式：

![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20L_i%7D%7B%5Cpartial%20W_j%7D%3D%5Cfrac%7B%5Cpartial%20L_i%7D%7B%5Cpartial%20q_i%7D%5Cfrac%7B%5Cpartial%20q_i%7D%7B%5Cpartial%20S_i%7D%5Cfrac%7B%5Cpartial%20S_i%7D%7B%5Cpartial%20W_j%7D)

观察shape：

对Wj求导后shape是(1，D)，后面三个分别是(1,C)，(C,C)，(C,D)，最终是(1,D)，记作(5)式：

![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20L_i%7D%7B%5Cpartial%20q_j%7D%3D%5B0%2C.....%2C-%5Cfrac%7B1%7D%7Bq_%7Biy_i%7D%7D%2C.....%2C0%5D)

记作(6)式：

![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20q_i%7D%7B%5Cpartial%20s_i%7D%3D%5Cleft%5C%7B%20%5Cbegin%7Bmatrix%7D%20.%20%26%20%26%20%26%20%26%20%26%20%5C%5C%20%26%20.%20%26%20%26%20%26%20%26%20%5C%5C%20%26%20%26.%20%26%20%26%20%26%20%5C%5C%20%26%20%26%20%26%20%5Cfrac%7B%5Cpartial%20q_%7Bim%7D%7D%7B%5Cpartial%20S_%7Bin%7D%7D%26%20%26%20%5C%5C%20%26%20%26%20%26%20%26%20.%26%20%5C%5C%20%26%20%26%20%26%20%26%20%26%20.%5C%5C%20%5Cend%7Bmatrix%7D%20%5Cright%5C%7D)

上面求导分为两种情况，记作(7)式：
$$
\begin{equation}
\frac{\partial q_{im}}{\partial S_{in}}=\left\{
\begin{array}{rcl}
m=n &  {\frac {e^{S_{im}}{\sum_ke^{S_{ik}}-e^{S_{im}}e^{S_{im}}}}{(\sum_k{e^S{ik})^2}}}=\frac {e^{S_{im}}}{\sum_k{e^S{ik}}}-(\frac {e^{S_{im}}}{\sum_k{e^S{ik}}})^2=q_{im}(1-q_{im})\\
m\neq n & \frac {0-e^{S_{in}}e^{S_{im}}}{(\sum_k{e^S{ik})^2}}=-q_{in}q_{im}
\end{array} \right.
\end{equation}
$$
Si表示S矩阵中每一行数据，那Sj对Wj求导如下：

现在取X矩阵第一行[X11,X12,.....X1n]

取W矩阵第一列[W11,W21....Wn1]

X与W矩阵相乘得S矩阵，上面X第一行与W第一列相乘得到S矩阵第一个元素，记作S01，同理我们可以得到S矩阵每一行得所有元素，分别为[Si1,Si2,.....,SiC] (i代表是哪一行)。

Wj代表W矩阵得列向量，每一列为Wj，第一列W1，后面依此类推！

那么我们现在来分析一下Si对Wj求导，这里推导：

对于最上面wj代表行向量，如下面所示是W矩阵(D,C)表示：记作(8)式：

![](https://latex.codecogs.com/gif.latex?%5Cleft%5C%7B%20%5Cbegin%7Bmatrix%7D%20...%26..%26...%26...%5C%5C%20w_1%26w_2%26...%26w_C%5C%5C%20...%26...%26...%26...%5C%5C%20...%26..%26...%26...%20%5Cend%7Bmatrix%7D%20%5Cright%5C%7D)

回顾一下(1)式，那么W转置得矩阵(C,D)则为：记作(9)式：

![](https://latex.codecogs.com/gif.latex?%5Cleft%5C%7B%20%5Cbegin%7Bmatrix%7D%20.%26.%26...%26w_1%5ET%26...%26.%5C%5C%20.%26.%26...%26.%26...%26.%5C%5C%20.%26.%26...%26w_2%5ET%26...%26.%5C%5C%20.%26.%26...%26.%26...%26.%5C%5C%20.%26.%26...%26w_C%5ET%26...%26.%5C%5C%20%5Cend%7Bmatrix%7D%20%5Cright%5C%7D)

而X矩阵(N,D)则是：记作(10)式：
$$
\left\{
 \begin{matrix}
   x_1&...&x_D\\
   x_2&...&x_D\\
   ...&...&x_D\\
  x_N&...&x_D\\
  \end{matrix}
  \right\}
$$
而S矩阵(N,C)表示为(记作)：记作(11)式：
$$
\left\{
 \begin{matrix}
   x_1w_1&x_1w_2&...&x_1w_j&...&x_1w_C\\
   x_iw_1&x_iw_2&...&x_iw_j&...&x_iw_C\\
     ...&...&...&...&...&...\\
   ...&...&...&...&...&x_Nw_C
  \end{matrix}
  \right\}
$$
也就是，记作(12)式：：
$$
\left\{
 \begin{matrix}
   s_1w_1&s_1w_2&...&s_1w_j&...&s_1w_C\\
   s_iw_1&s_iw_2&...&s_iw_j&...&s_1w_C\\
   ...&...&...&...&...&...\\
   s_Nw_1&s_Nw_2&...&s_Nw_j&...&s_Nw_C
  \end{matrix}
  \right\}
$$
S1表示第一行，Si表示第i行

现在回到求导，那么当Si对Wj进行求导得时候，我们从列向量表示得S矩阵(12)与原始矩阵S(11)相比较，我们知道，Si对wj求导为xi，其余全为0，得到下面结果，记作(13)式(C,D)：
$$
\left\{
 \begin{matrix}
  .. &..&..&0 & .. &..\\
  .. &..&....&.& .. &..\\
  .. &..&....&xi &..&..\\
   .. &..&....&.& .. &..\\
  .. &..&....&0& .. &..
  \end{matrix}
  \right\}
$$


带入链式求导法则，得到：


$$
\frac{\partial L_i}{\partial W_j}=\frac{\partial L_i}{\partial q_i}\frac{\partial q_i}{\partial S_i}\frac{\partial S_i}{\partial W_j}=[0，.....，-\frac{1}{q_{iy_i}}，.....，0]\left\{
 \begin{matrix}
   . &  & & & & \\
   & . & & & &  \\
    &   &. & & & \\
    &  & & \frac{\partial q_{im}}{\partial S_{in}}& & \\
   &  & & & .&  \\
    &   & & & & .\\
  \end{matrix}
  \right\}\left\{
 \begin{matrix}
  .. &..&..&0 & .. &..\\
  .. &..&....&.& .. &..\\
  .. &..&....&xi &..&..\\
   .. &..&....&.& .. &..\\
  .. &..&....&0& .. &..
  \end{matrix}
  \right\}\\=[q_{i1},q{i2},...,q_{i_{y_i}}-1,...q_{ic}]\left\{
 \begin{matrix}
  .. &..&..&0 & .. &..\\
  .. &..&....&.& .. &..\\
  .. &..&....&xi &..&..\\
   .. &..&....&.& .. &..\\
  .. &..&....&0& .. &..
  \end{matrix}
  \right\}=x_iq_{ij}\begin{equation}\left\{
\begin{array}{rcl}
j\neq y_i &  q_{ij}\\
j=y_i & q_{i_{y_i}}-1
\end{array} \right.
\end{equation}
$$


> 梯度实现

在上述交叉熵下面添加如下代码即可！

```python
# 计算梯度
for j in range(num_class):
  if j!=y[i]:
    dw[i,j]+=score[j]*X[i]
  else:
    dw[i,j]+=(score[j]-1)*X[i]
```
















































































































































































