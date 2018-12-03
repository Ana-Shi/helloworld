# 全连接神经网络(上)

## 0.说在前面

【**回顾与任务**】

在上次作业中，已经实现了两层神经网络，但是有些问题，比如程序不够模块化，耦合度不高等问题，所以本节引出神经网络的层与层结构。本节主要实现一种模块化的神经网络架构，将各个功能封装为一个对象，包括全连接层对象，仿射层，Relu层等，在各层对象的前向传播函数中，将由上一层传来的数据和本层的相关参数，经过本层的激活函数，生成输出值，并将在后面反向传播需要滴额参数，进行缓存处理，将根据后面层次的提取与缓存值计算本层各参数的梯度，从而实现反向传播。

有关更多内容，请关注微信公众号：guangcity

![](https://github.com/Light-City/images/blob/master/wechat.jpg?raw=true)

## 1.仿射层

【**仿射层前向传播**】

前向传播比较简单，就是直接计算，注意矩阵的运算即可。

- 目标：计算实现一个仿射层的前向传播
- 输入

```python
- x:  (N, d_1, ..., d_k)
- w:  (D, M)
- b:  (M,)
```

- 返回

```
- out:  (N, M)
- cache: (x, w, b)
```

- 实现

改变x的shape，使得满足矩阵乘法，这里-1表示模糊匹配，b使用了广播机制。cache存储当前层的反向传播计算参数，out存储输出值。

```python
def affine_forward(x, w, b):
    out = None
    out = np.dot(x.reshape(x.shape[0],-1),w)+b
    cache = (x, w, b)
    return out, cache
```

【**仿射层反向传播**】

- 目标：计算仿射层的后向传播
- 输入

```python
- dout:  (N, M)
- cache: 
    x:  (N, d_1, ... d_k)
    w:  (D, M)
    b:  (M,)
```

- 返回

```python
- dx:  (N, d1, ..., d_k)
- dw:  (D, M)
- db:  (M,)
```

首先获得上面前向传播的输出值与cache，紧接着计算反向传播。

- 实现

cache解开得到前面仿射层的前向传播参数，接着计算梯度即可！

```python
def affine_backward(dout, cache):
    x, w, b = cache
    dx, dw, db = None, None, None
    db = np.sum(dout,axis=0)
    dw = np.dot(x.reshape(x.shape[0],-1).T,dout)
    dx = np.dot(dout,w.T).reshape(x.shape)
    return dx, dw, db
```

## 2.Relu层

【**Relu前向传播**】

- 目标：计算Relu的前向传播
- 输入：

```python
x:  任意shape的输入
```

- 返回

```python
- out: 输出同x一样的shape
- cache: x
```

- 实现

上面目标很明确，这里直接来实现，不多解释，这里用到了一个布尔矩阵运算，如果觉得疑惑，请看作业详解knn中的解释！

```python
def relu_forward(x):
    out = None
    out = x*(x>0)
    cache = x
    return out, cache
```

【**Relu反向传播**】

- 目标：计算Relu的后向传播
- 输入

```python
- dout:  任何shape的前向输出(这里疑惑的看英文原文)
- cache：同dout相同shape的x 
```

- 实现

Relu只有矩阵中大于0的数有效，所以x>0筛选得出一个布尔矩阵，直接相乘就是最后的结果。因为如果x<0，求导肯定为0，所以有效值，就是x>0的结果！

```python
def relu_backward(dout, cache):
    dx, x = None, cache
    dx = dout*(x>0)
    return dx
```



## 3.仿射层与Relu层组合

【**前向传播**】

- 目标：完成仿射层与Relu层组合
- 输入：

```
- x: 仿射层的输入
- w, b: 仿射层的权重
```

- 返回

```python
- out: ReLU层输出
- cache: 后向传播的缓存
```

- 实现

```python
def affine_relu_forward(x, w, b):
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache
```

【**反向传播**】

- 目标：实现反向传播
- 输入：

```
- dout
- cache
```

- 实现

直接调用刚才的方法。

```python
def affine_relu_backward(dout, cache):
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db
```

## 4.重新实现两层神经网络

【**类封装**】

有关svm与softmax略过，具体看前面内容！

- 封装
- 目标：实现affine - relu - affine - softmax架构
- 输入：

```python
- input_dim:  输入层尺寸
- hidden_dim: 隐藏层尺寸
- num_classes: 类别数
- dropout: 随机失活强度 0～1
- weight_scale: 
- reg: 
```

- 实现：

封装全局参数

```python
class TwoLayerNet(object):
    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        self.params = {}
        self.reg = reg
        self.params['W1'] = np.random.n·ormal(0, weight_scale, (input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.params['b2'] = np.zeros(num_classes)
```

【**损失函数**】

- 输入：

```python
- X:  (N, d_1, ..., d_k)
- y:  (N,) 
```

- 返回

```python
返回:
If y is None, 运行 test-time forward
返回:
- scores: (N, C) 
If y is not None, 运行 training-time forward 和 backward pass
返回:
- loss
- grads
```

- 实现

```python
def loss(self, X, y=None):
    scores = None   
    # 直接调用relu函数
    affine_relu_out, affine_relu_cache = affine_relu_forward(X, self.params['W1'], self.params['b1'])
    # relu前向传播
    affine2_out, affine2_cache = affine_forward(affine_relu_out, self.params['W2'], self.params['b2']) 
    scores = affine2_out
    # If y is None then we are in test mode so just return scores
    if y is None:
        return scores
    loss, grads = 0, {}
    # softmax获取损失值与得分
    loss, dscores = softmax_loss(scores, y)
    # 添加正则项
    loss += 0.5 * self.reg*(np.sum(self.params['W1']* self.params['W1']) + np.sum(self.params['W2']* self.params['W2']))
    # 仿射层反向传播
    affine2_dx, affine2_dw, affine2_db = affine_backward(dscores, affine2_cache)
    grads['W2'] = affine2_dw + self.reg * self.params['W2']   
    grads['b2'] = affine2_db
    # relu层反向传播
    affine1_dx, affine1_dw, affine1_db = affine_relu_backward(affine2_dx, affine_relu_cache)
    grads['W1'] = affine1_dw + self.reg * self.params['W1']
    grads['b1'] = affine1_db
    return loss, grads
```



## 5.Solver训练

【**概要**】

使用这个训练之前，需要补充optim.py！

此文件实现了常用的各种一阶更新规则用于训练神经网络。每个更新规则接受当前权重和相对于那些权重的损失梯度并产生下一组权重！

【**sgd**】

> 公式

![](https://github.com/Light-City/images/blob/master/sgd.jpg?raw=true)

这个被称为朴素sgd(Vanilla SGD)

公式中四个参数分别对应为：下一次的权重w，当前权重w，学习率，当前权重的梯度！

> 实现

```python
def sgd(w, dw, config=None):
    '''
    - learning_rate: Scalar learning rate.
    '''
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    w -= config['learning_rate'] * dw
    return w, config
```

【**sgd_momentum**】

> 公式

![](https://github.com/Light-City/images/blob/master/mom.jpg?raw=true)

这个被称为结合动量的sgd(最常用)。

阿尔法代表学习率！



> 实现

```python
def sgd_momentum(w, dw, config=None):
    '''
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.
    '''
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))
    next_w = None
    next_w = w
    # 更新v
    v = config['momentum']* v - config['learning_rate']*dw
    # 更新w
    next_w +=v
    # 保存v
    config['velocity'] = v
    return next_w, config
```

【**RMSProp**】

> 公式

![](https://github.com/Light-City/images/blob/master/rms.jpg?raw=true)

> 实现

```python
def rmsprop(w, dw, config=None):
    '''
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    '''
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(w))
    next_w = None
    # 更新vt
    config['cache'] = config['decay_rate']*config['cache']+(1-config['decay_rate'])*(dw*dw)
    # 更新w
    next_w = w-config['learning_rate']* dw / (np.sqrt(config['cache'])+config['epsilon'])
    return next_w, config
```

【**adam**】

> 公式

![](https://github.com/Light-City/images/blob/master/adam.jpg?raw=true)

> 实现

```python
def adam(w, dw, config=None):
    """
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(w))
    config.setdefault('v', np.zeros_like(w))
    config.setdefault('t', 0)
    next_w = None
    config['t']+=1 
    # 对应公式：更新公式中的mt
    config['m'] = config['beta1']*config['m'] + (1- config['beta1'])*dw
    # 对应公式更新公式中的nt
    config['v'] = config['beta2']*config['v'] + (1- config['beta2'])*(dw**2)   
    # 对应公式：校正后的mt
    mb = config['m']/(1-config['beta1']**config['t'])
    # 对应公式：校正后的nt
    vb = config['v']/(1-config['beta2']**config['t'])
    # 更新w
    next_w = w -config['learning_rate']* mb / (np.sqrt(vb) + config['epsilon'])
    return next_w, config
```

【**Solver训练**】

```python
model = TwoLayerNet()
solver = None
model = TwoLayerNet(hidden_dim=100, reg=0.2)
solver = Solver(model, data,
                update_rule='sgd',
                optim_config={
                  'learning_rate': 1e-3,
                },
                lr_decay=0.95,
                num_epochs=20, batch_size=500,
                print_every=500)
solver.train()
```

训练结果：

![](https://github.com/Light-City/images/blob/master/train_res1.jpg?raw=true)

![](https://github.com/Light-City/images/blob/master/train_res2.jpg?raw=true)

