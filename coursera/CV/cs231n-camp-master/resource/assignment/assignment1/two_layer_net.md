

# Softmax及两层神经网络

## 0.说在前面

今天重点来推导Softmax及两层神经网络及代码实现，下面一起来实战吧！

有关更多内容，请关注微信公众号：guangcity

![](https://github.com/Light-City/images/blob/master/wechat.jpg?raw=true)



## 1.Softmax向量化

### 1.1 Softmax梯度推导

首先来给出Loss的公式

![](https://latex.codecogs.com/gif.latex?L%3D%5Cfrac%201N%5Csum_%7Bi%3D1%7D%5ENL_i+%5Csum_kW_k%5E2)

data loss+regularization！

推导：

X矩阵是(N,D)，W矩阵是(D,C)，S矩阵是(N,C)，S矩阵中每一行是Li，那么XW=S表示如下公式(1)所示：

![](https://latex.codecogs.com/gif.latex?%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20..%20%26..%26x_1%20%26%20..%20%26..%5C%5C%20..%20%26..%26.%26%20..%20%26..%5C%5C%20..%20%26..%26x_i%20%26..%26..%5C%5C%20..%20%26..%26.%26%20..%20%26..%5C%5C%20..%20%26..%26x_N%26%20..%20%26..%20%5Cend%7Bmatrix%7D%5Cright%5C%7D%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20..%20%26..%26.%26%20..%20%26..%5C%5C%20..%20%26..%26.%26%20..%20%26..%5C%5C%20w_1%20%26..%26w_j%20%26..%26w_c%5C%5C%20..%20%26..%26.%26%20..%20%26..%5C%5C%20..%20%26..%26.%26%20..%20%26..%20%5Cend%7Bmatrix%7D%5Cright%5C%7D%3D%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20..%20%26..%26s_1%20%26%20..%20%26..%5C%5C%20..%20%26..%26.%26%20..%20%26..%5C%5C%20..%20%26..%26s_i%20%26..%26..%5C%5C%20..%20%26..%26.%26%20..%20%26..%5C%5C%20..%20%26..%26s_N%26%20..%20%26..%20%5Cend%7Bmatrix%7D%5Cright%5C%7D)

L对W求导，最后的矩阵维度为W的维度，那么L对W求导维度为(D,C)，而L对S的求导维度为(N,C)，S对W的求导维度为(N,D)或者(D,N)，根据**维度相容**来选择，如果X与W均是一维的那么就是X，否则就是X转置！下面的式子记作(2)式：

![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%7D%3D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20S%7D%5Cfrac%7B%5Cpartial%20S%7D%7B%5Cpartial%20W%7D%3DX%5ET%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20S%7D)

X转置后维度为(D,N)，而L对S求导的维度为(N,C)，此时可以相乘，否则不能乘！

L对Si求导，我们知道L1只与S1有关，推出Li只与Si有关！下面的式子记作(3)式：

![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20S%7D%3D%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20..%20%26..%26%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20S_1%7D%20%26%20..%20%26..%5C%5C%20..%20%26..%26.%26%20..%20%26..%5C%5C%20..%20%26..%26%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20S_i%7D%20%26..%26..%5C%5C%20..%20%26..%26.%26%20..%20%26..%5C%5C%20..%20%26..%26%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20S_N%7D%26%20..%20%26..%20%5Cend%7Bmatrix%7D%5Cright%5C%7D%3D%5Cfrac%201N%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20..%20%26..%26%5Cfrac%7B%5Cpartial%20L_1%7D%7B%5Cpartial%20S_1%7D%20%26%20..%20%26..%5C%5C%20..%20%26..%26.%26%20..%20%26..%5C%5C%20..%20%26..%26%5Cfrac%7B%5Cpartial%20L_i%7D%7B%5Cpartial%20S_i%7D%20%26..%26..%5C%5C%20..%20%26..%26.%26%20..%20%26..%5C%5C%20..%20%26..%26%5Cfrac%7B%5Cpartial%20L_N%7D%7B%5Cpartial%20S_N%7D%26%20..%20%26..%20%5Cend%7Bmatrix%7D%5Cright%5C%7D)

紧接着，我们将Li对Si求导拆分成对q求导，在由q对S求导，这里的推论结果，直接使用上次推出的结果，带入就是下面的额式子(记作(4)式)：

![](https://github.com/sharedeeply/cs231n-camp/blob/master/resource/img/snn_q.png?raw=true)

完成(2)式得，记作(5)式：

![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%7D%3D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20S%7D%5Cfrac%7B%5Cpartial%20S%7D%7B%5Cpartial%20W%7D%3DX%5ET%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20S%7D%3DX%5ET%5Cfrac%201N%5Csum_i%5EN%5Cfrac%7B%5Cpartial%20L_i%7D%7B%5Cpartial%20S_i%7D%3DX%5ET%5Cfrac%201N%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20q_%7B11%7D%20%26%20q_%7B22%7D%20%26%20q_%7B1_%7By_1%7D%7D-1%20%26%20...%20%26%20q_%7B1C%7D%5C%5C%20q_%7B21%7D%20%26%20q_%7B22%7D%20%26%20q_%7B2_%7By_2%7D%7D-1%20%26%20...%20%26%20q_%7B2C%7D%5C%5C%20q_%7Bi1%7D%20%26%20q_%7Bi2%7D%20%26%20q_%7Bi_%7By_i%7D%7D-1%20%26%20...%20%26%20q_%7BiC%7D%5C%5C%20.%20%26%20...%20%26%20...%20%26%20...%20%26%20.%5C%5C%20.%20%26%20...%20%26%20...%20%26%20...%20%26%20.%5C%5C%20q_%7BN1%7D%20%26%20q_%7BN2%7D%20%26%20q_%7BN_%7By_N%7D%7D-1%20%26%20...%20%26%20q_%7BNC%7D%20%5Cend%7Bmatrix%7D%5Cright%5C%7D)

### 1.2 Softmax向量化实现

具体实现的流程解释看代码注释！

```python
def softmax_loss_vectorized(W, X, y, reg):
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_class = W.shape[1]
    scores = X.dot(W)  # N*C
    # np.max后会变成一维，可设置keepdims=True变为二维(N,1)
    # 防止指数爆炸
    scores-=np.max(scores,axis=1,keepdims=True)
    # 取指数
    scores=np.exp(scores)
    # 计算softmax
    scores/=np.sum(scores,axis=1,keepdims=True)
    # ds表示L对S求导
    ds = np.copy(scores)
    # qiyi - 1
    ds[np.arange(num_train), y] -= 1
    dW = np.dot(X.T, ds)
    loss = scores[np.arange(num_train), y]
    # 计算Li
    loss =-np.log(loss).sum()
    # 计算所有loss除以N
    loss /= num_train
    # ds矩阵没有除以N，所以在这里补上，最后除以N，具体看(5)式
    dW /= num_train
    # 计算最终的大L
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    return loss, dW
```



## 2.两层神经网络

### 2.1 反向传播推导

![](https://github.com/sharedeeply/cs231n-camp/blob/master/resource/img/two_layer.png?raw=true)

### 2.2 两层神经网络实现

> 计算前向传播

前向传播可以看上面手推图结构！

```python
scores = None
s1 = np.dot(X, W1) + b1
# (N,H)
s1_relu = (s1 > 0) * s1
scores = np.dot(s1_relu, W2) + b2
if y is None:
    return scores
```

> 计算损失函数

这里计算损失与softmax一致，可以参看上面的！

```python
# Compute the loss
loss = None
# 防止指数爆炸
scores -= np.max(scores, axis=1, keepdims=True)
# 取指数
scores = np.exp(scores)
# 计算softmax
scores /= np.sum(scores, axis=1, keepdims=True)
loss = -np.log(scores[np.arange(N), y]).sum()
loss /= N
loss += reg * np.sum(W1 * W1)
loss += reg * np.sum(W2 * W2)
```
> 计算反向传播

具体推导看上面手推图！

这里将上面的关键点提出来，ds2表示的是dl对ds2求导，ds1表示dl对ds1求导！其余的一致！

```python
grads = {}
ds2 = np.copy(scores)
# qiyi - 1
ds2[np.arange(N), y] -= 1
grads['W2'] = np.dot(s1_relu.T, ds2) / N + 2 * reg * W2
# b2的shape=(N,C)广播机制
# (1,C)
# 这里除以N是因为ds的时候没有除以N，所以最后就得除以N，后面相同！
grads['b2'] = np.sum(ds2, axis=0) / N
ds1 = np.dot(ds2, W2.T)
# relu函数
ds1 = (s1 > 0) * ds1
grads['W1'] = np.dot(X.T, ds1) / N + 2 * reg * W1
grads['b1'] = np.sum(ds1, axis=0) / N
```
> 随机选择数据集batch_size大小

`train`方法中添加：

```python
num_random = np.random.choice(np.arange(num_train), batch_size)
X_batch = X[num_random, :]
y_batch = y[num_random]
```

> 计算损失与梯度

`train`方法中添加：

```python
loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
loss_history.append(loss)
```

> 更新w与b

`train`方法中添加：

```python
self.params['W1'] -= learning_rate * grads['W1']
self.params['W2'] -= learning_rate * grads['W2']
self.params['b1'] -= learning_rate * grads['b1']
self.params['b2'] -= learning_rate * grads['b2']
```

> 预测结果

```python
output = np.maximum(X.dot(self.params['W1']) + self.params['b1'], 0).dot(self.params['W2'])+self.params['b2']
y_pred = np.argmax(output, axis=1)
```

