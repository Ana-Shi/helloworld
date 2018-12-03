# SVM梯度推导

## 0.说在前面

今天重点来推导梯度及代码实现，下面一起来实战吧！

有关更多内容，请关注微信公众号：guangcity

![](https://github.com/Light-City/images/blob/master/wechat.jpg?raw=true)

## 1.梯度推导

> loss fuction

SVM损失函数想要SVM在正确分类上的得分始终比不正确的得分高出一个边界值！

下面来进行梯度下降的推导！

> loss vector

数学推导如下：

X表示如下(1式)：

![](https://latex.codecogs.com/gif.latex?%5Cleft%5C%7B%20%5Cbegin%7Bmatrix%7D%20...%20%26%20x_%7B1%7D%20%26%20...%20%5C%5C%20...%20%26%20x_%7B2%7D%20%26%20...%20%5C%5C%20%26%20.%20%26%5C%5C%20%26%20.%20%26%5C%5C%20...%20%26%20x_%7BN%7D%20%26%20...%20%5Cend%7Bmatrix%7D%20%5Cright%5C%7D%20%5Ctag%7B1%7D)



W表示如下(2式)：

![](https://latex.codecogs.com/gif.latex?%5Cleft%5C%7B%20%5Cbegin%7Bmatrix%7D%20.%20%26%20.%20%26%20...%20%26%20.%5C%5C%20.%26%20.%20%26%20...%20%26%20.%20%5C%5C%20w_%7B1%7D%20%26%20w_%7B2%7D%20%26%20...%20%26%20w_%7BC%7D%20%5C%5C%20.%20%26%20.%20%26%20...%20%26%20.%5C%5C%20.%26%20.%20%26%20...%20%26%20.%20%5C%5C%20%5Cend%7Bmatrix%7D%20%5Cright%5C%7D%20%5Ctag%7B2%7D)

S表示如下(3式)：

![](https://latex.codecogs.com/gif.latex?%5Cleft%5C%7B%20%5Cbegin%7Bmatrix%7D%20s_%7B11%7D%20%26%20s_%7B12%7D%20%26%20...%20%26%20s_%7B1C%7D%20%5C%5C%20s_%7B21%7D%20%26%20s_%7B22%7D%20%26%20...%20%26%20s_%7B2C%7D%5C%5C%20.%20%26%20.%20%26%20...%20%26%20.%5C%5C%20.%20%26%20.%20%26%20...%20%26%20.%5C%5C%20.%20%26%20.%20%26%20...%20%26%20.%5C%5C%20s_%7BN1%7D%20%26%20s_%7BN2%7D%20%26%20...%20%26%20s_%7BNC%7D%20%5Cend%7Bmatrix%7D%20%5Cright%5C%7D%20%5Ctag%7B3%7D)


X的shape为NxD，W的shape为DxC，S的shape为NxC，

```
X*W=S
f(S)=L # f表示loss function
```

这里使用链式求导法对w进行求导(4式)：

![](https://latex.codecogs.com/gif.latex?%5Cfrac%20%7B%5Cpartial%20L%7D%20%7B%5Cpartial%20w%7D%3D%5Cfrac%7B%5Cpartial%20L%7D%20%7B%5Cpartial%20S%7D%5Cfrac%7B%5Cpartial%20S%7D%20%7B%5Cpartial%20w%7D%3D%7BX%5ET%7D%5Cfrac%7B%5Cpartial%20L%7D%20%7B%5Cpartial%20S%7D%20%3D%20%7BX%5ET%7D%5Cleft%5C%7B%20%5Cbegin%7Bmatrix%7D%20...%20%26%20%5Cfrac%20%7B%5Cpartial%20L%7D%20%7B%5Cpartial%20s_1%7D%20%26%20...%20%5C%5C%20...%20%26%20%5Cfrac%20%7B%5Cpartial%20L%7D%20%7B%5Cpartial%20s_1%7D%20%26%20...%20%5C%5C%20%26%20.%20%26%5C%5C%20%26%20.%20%26%20%5Cend%7Bmatrix%7D%20%5Cright%5C%7D)

由上面知道X的shape为NxD，由于L对S求导的shape为NxC，而NxD矩阵与NxC矩阵不能直接相乘，故要对X进行转置！

对(4)内部进行求导拆分，如(5)式

s1只跟L1有关，si只跟Lj有关，于是求和可以去掉，转化为后面那个Lj对Sj求导！

(5式)

![](https://latex.codecogs.com/gif.latex?%5Cfrac%20%7B%5Cpartial%20L%7D%20%7B%5Cpartial%20S_j%7D%3D%5Csum_%7Bi%7D%5Cfrac%7B%5Cpartial%20L_i%7D%20%7B%5Cpartial%20S_j%7D%3D%5Cfrac%7B%5Cpartial%20L_j%7D%20%7B%5Cpartial%20S_j%7D%20%3D%20%5B%7B%20%5Cbegin%7Bmatrix%7D%20%5Cfrac%20%7B%5Cpartial%20L_%7Bj%7D%7D%20%7B%5Cpartial%20s_%7Bj1%7D%7D%20%26%20%5Cfrac%20%7B%5Cpartial%20L_%7Bj%7D%7D%20%7B%5Cpartial%20s_%7Bj2%7D%7D%20%26%20...%20%26%20%5Cfrac%7B%5Cpartial%20L_%7Bj%7D%7D%20%7B%5Cpartial%20s_%7Bjc%7D%7D%20%5Cend%7Bmatrix%7D%20%7D%5D)

对Lj求和展开(6式)

![](https://latex.codecogs.com/gif.latex?L_j%3D%5Csum_%7Bk%5Cneq%7By_j%7D%7Dmax%280%2CS_%7Bjk%7D-S_%7Bj%2Cy_j%7D+%5CDelta%29%3Dmax%280%2CS_%7Bj1%7D-S_%7Bjy_j%7D+%5CDelta%29+...max%280%2CS_%7Bjc%7D-S_%7Bjy_j%7D+%5CDelta%29%20%5Ctag%7B6%7D)

将Lj式子带入（5）式，我们知道，与Sj1相关的只有第一项，那么就是1，当max函数算出的值大于0，对于Sjk(k不等于yj)的时候，求导为1，否则为0；而对于Sjyj的时候，也就是k等于yj的时候求导，当max函数算出的值大于0，求导为-1，否则为0，将所有的值相加就是最后的k=yj求导结果。

## 2.实现

> loss native

```python
def svm_loss_vectorized(W, X, y, reg):
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    # (N,D)
    # N
    num_train = X.shape[0]
    # (D,C)
    # C
    num_classes = W.shape[1]
    # (N,C)
    scores = X.dot(W)
    # 获取每个样本对应label的得分
    # (N,1)
    y_score =scores[np.arange(num_train),y].reshape(-1,1)
    mask = (scores-y_score+1)>0
    scores = (scores-y_score+1)*mask
    loss =(np.sum(scores)-num_train*1)/num_train
    loss += reg * np.sum(W * W)  
    # 初始化ds,
    ds = np.ones_like(scores)
    # 有效的score梯度为1，无效的为0
    ds*=mask
    # 去掉k=yj的情况
    ds[np.arange(num_train),y]=-1*(np.sum(mask,axis=1)-1)
    # 对应公式(4)
    dW=np.dot(X.T,ds)/num_train
    dW+=2*reg*W
    return loss, dW

```

这里的代码引自训练营老师代码，这里写的很精辟，非常值得学习，特别是对于这个逻辑处理，下面来深入分析一下。

> 模拟实验

首先是找到有效分数，也就是上述max()函数大于0与小于0的情况，这里直接通过先判断，这里来模拟一下操作：

首先定义两个二维数组，分别是x1与x2：x1表示上面公式推导的(3)式子，S矩阵，也就是通过X与W相乘后的矩阵！x2表示预测结果为真实label的yj。

下面一起来学习一下模拟操作以及老师在这里的运算精髓！

**In**

```python
x1 = np.array(np.arange(6)).reshape(3,2)
x2 = np.array([2,3,3]).reshape(-1,1)
x1
x2
```

**Out**

```python
array([[0, 1],
       [2, 3],
       [4, 5]])
array([[2],
       [3],
       [3]])
```

**In**

```python
mask = x1-x2+1>0
```

通过这个操作可以得到什么？

**Out**

```python
array([[False, False],
       [False,  True],
       [ True,  True]])
```

发现没，得到了一个布尔类型的同shape多维数组，那么是不是可以这样说，思路就是首先通过得到有效无效分数的布尔高维数组，有效就是True，无效就是False，在运算的时候，直接可以将False看作0，True看作1。

那么我们接下来计算真实的得分：

**In**

```python
scores=(x1-x2+1)*mask
scores
```

**Out**

```python
array([[0, 0],
       [0, 1],
       [2, 3]])
```

这个就是我们期望通过下面公式得到的结果！

![](https://latex.codecogs.com/gif.latex?max%280%2CS_%7Bjc%7D-S_%7Bjy_j%7D+1%29)

这里实现的方法非常巧妙！！！

这样做的好处是可以避免循环及使用max函数！！

上述的代码等价于：

**In**

```python
for i in range(x1.shape[0]):
    for j in range(x1.shape[1]):
        x1[i][j]=max(0,x1[i][j]-x2[i][0]+1)
```

**Out**

```python
array([[0, 0],
       [0, 1],
       [2, 3]])
```

> 注意点

由于上述公式(6)在计算得分时候不涉及yj也就是预测为真正label这一项，而在计算的时候，实际上把所有的都计算了，所以得减去这一项，对于每一行操作都多算了一个：

![](https://latex.codecogs.com/gif.latex?max%280%2CS_%7Bjy_j%7D-S_%7Bjy_j%7D+1%29)

总共num_train行，所以是num_train*1个！

```python
loss =(np.sum(scores)-num_train*1)/num_train
```

另外就是在计算ds的时候，这个ds就是上述(5)式，L对S求导，最终求导结论就是：

(1)当对Sjk求导时(k不等于yj)，若为有效得分，则为1，否则为0；

(2)当对Sjk求导时(k等于yj)时，若为有效得分，则为多个-1，否则为0；

第(1)点主要说的是下面这个式子当中的Sjk，第(2)主要说的是下面这个式子中的Sjyj求导，因为是多个max函数累加，那么对于Sjyj求导的话是每一个max都有一项，所以**如果max得分是正数，则表示求导结果是-1**，将**多个求导的-1叠加**就是**最后对Sjyj的求导总和**；而对于Sjk求导，虽然是多个max，以求Sj1为例，**只有一个max中存在Sj1，所以总的求导要么为1，要么为0**。

![](https://latex.codecogs.com/gif.latex?max%280%2CS_%7Bjk%7D-S_%7Bjy_j%7D+1%29)

所以最后的L对S求导结果用代码实现就是：

```python
# 有效的score梯度为1，无效的为0
ds*=mask
# 去掉k=yj的情况
ds[np.arange(num_train),y]=-1*(np.sum(mask,axis=1)-1)
```
上面说了mask是将预测为yj的也算进去了，对于当碰到下面这个式子：

![](https://latex.codecogs.com/gif.latex?max%280%2CS_%7Bjy_j%7D-S_%7Bjy_j%7D+1%29)

由于每一个mask中都多算了一个这个式子，而这个式子总是有效的！！！所以直接减去这次的结果就是去掉k等于yj的情况！



## 







