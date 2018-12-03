# 全连接神经网络(下)

## 0.说在前面

我们继续来上次cs231n的assignment2的全连接神经网络第二篇。这一篇则重点研究构建任意层数的全连接网络！下面我们一起来实战吧！

有关更多内容，请关注微信公众号：guangcity

![](https://github.com/Light-City/images/blob/master/wechat.jpg?raw=true)

## 1.Batch Normalization

### 1.1 什么是BN?

什么是Batch Normalization，以及相关的前向传播，反向传播推导，这里给出一个大佬的网址，大家可以自行mark！

[Understanding the backward pass through Batch Normalization Layer](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html)

简单来说，Batch Normalization就是在每一层的wx+b和f(wx+b)之间加一个归一化。

什么是归一化，这里的归一化指的是：将wx+b归一化成：均值为0，方差为1！

下面给出Batch Normalization的算法和反向求导公式，下图来自于网上上述链接~

![](https://github.com/Light-City/images/blob/master/bn_al.png?raw=true)

### 1.2 前向传播

前向与后向传播均在layes.py文件内！

其实这里比较好写，原因在于注释提示了很多比如注释里面的：

```python
running_mean = momentum * running_mean + (1 - momentum) * sample_mean
running_var = momentum * running_var + (1 - momentum) * sample_var
```

**输入输出：**

```
输入:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

返回元组:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
```

**完整实现：**

相关公示的注释已经写上，对上述的算法进行实现即可！

```python
def batchnorm_forward(x, gamma, beta, bn_param):
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        # mini-batch mean miu_B (1,D)
        sample_mean = np.mean(x,axis=0,keepdims=True)
        # miin-batch variance sigema_square (1,D)
        sample_var = np.var(x,axis=0,keepdims=True)
        # normalize (N,D)
        x_normalize = (x-sample_mean)/np.sqrt(sample_var+eps)
        # scale and shift
        out = gamma*x_normalize+beta
        cache=(x_normalize,gamma,beta,sample_mean,sample_var,x,eps)
        # update
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
    elif mode == 'test':
        x_normalize = (x-running_mean)/np.sqrt(running_var+eps)
        out = gamma*x_normalize+beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache
```

### 1.3 反向传播

反向传播很重要，而在assignment1中对两层神经网络进行手推，这里是一样的原理，由于自己写的有点乱，就不放手推了，给出网上的推导：

![](https://github.com/Light-City/images/blob/master/bn_qiudao.png?raw=true)

**输入输出：**

```python
输入:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

返回元组:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
```

**完整实现：**

这里建议将上述算法与反向传播公式联系起来一起推，最好手推，有一个重要点提一下，就是在对x求导的时候，是层层嵌套，所以采用算法当中的分治法解决，分为多个子问题，链式推导，方便简单，而且不容易出错！

```python
def batchnorm_backward(dout, cache):
    dx, dgamma, dbeta = None, None, None
    x_normalized, gamma, beta, sample_mean, sample_var, x, eps = cache
    N, D = x.shape
    dx_normalized = dout * gamma       # [N,D]
    x_mu = x - sample_mean             # [N,D]
    sample_std_inv = 1.0 / np.sqrt(sample_var + eps)    # [1,D]
    dsample_var = -0.5 * np.sum(dx_normalized * x_mu, axis=0, keepdims=True) * sample_std_inv**3
    dsample_mean = -1.0 * np.sum(dx_normalized * sample_std_inv, axis=0, keepdims=True) - 2.0 * dsample_var * np.mean(x_mu, axis=0, keepdims=True)
    dx1 = dx_normalized * sample_std_inv
    dx2 = 2.0/N * dsample_var * x_mu
    dx = dx1 + dx2 + 1.0/N * dsample_mean
    dgamma = np.sum(dout * x_normalized, axis=0, keepdims=True)
    dbeta = np.sum(dout, axis=0, keepdims=True)
    return dx, dgamma, dbeta
```

## 2.Dropout

### 2.1 什么是Dropout？

Dropout可以理解为遗抑制过拟合的一种正规化手段！在训练过程中，对每个神经元，以概率p保持它的激活状态。下面给出dropout的示意图：

![](https://github.com/Light-City/images/blob/master/dropout.png?raw=true)

回答先图b与图a明显的区别是，指向变少了，也就是去掉了很多传递过程，但在实际中不经常用，因为容易去掉一些关键信息！

### 2.2 前向传播

前向与反向传播在layers.py文件中！

在注释中提到了cs231n的一个关键点，大家可以去下面链接去看什么是dropout:

[cs231n直通点](http://cs231n.github.io/neural-networks-2/#reg)

**输入输出**

```python
输入:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

 输出:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
```

**完整实现**

具体实现只需要记住一句话，以某一概率失活！！！也就是让当前的数据乘以每个数据的失活概率即可！

```python
def dropout_forward(x, dropout_param):
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])
    mask = None
    out = None
    if mode == 'train':
        mask = (np.random.rand(*x.shape) < p)   #以某一概率随机失活
        out = x * mask
    elif mode == 'test':
        out=x
    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)
    return out, cache
```

### 2.3 反向传播

**输入输出：**

```python
输入:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
输出：
    - dx
```

**完整实现：**

实现就是直接上层的梯度乘以当前的梯度，上层梯度为dout，当前梯度为存储的mask。

```python
def dropout_backward(dout, cache):
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        dx = dout * mask
    elif mode == 'test':
        dx = dout
    return dx
```

## 3.任意隐藏层数的全连接网络

对fc_net.py进行修改！

对于这一块填写，之前一直有点不懂，还好今天重新看了一下注释，觉得很清楚了，建议都去看看注释的todo或者解释，很详细！！！

以这个为例：

首先我们可以看到所构建的全连接网络结构为：

网络的层数为L层，L-1表示重复{blok}L-1次，注释中都有的!

```python
{affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax
```

**输入输出：**

为了保持原文意思，这里没有翻译出来，大家克服一下，看英文，如果不懂可以留言！

```python
  初始化一个新的全连接网络.
        输入:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
```

下面两行#号中间为填写内容！我们所实现的目标大家可以看TODO，里面说的很详细，我简单说一下，就是来存储w与b，而这个存储的作用，则会在后面的loss用到！

```python
class FullyConnectedNet(object):
    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        ############################################################################
        num_neurons = [input_dim] + hidden_dims + [num_classes]
        # 看一开始的时候注释就说了L-1次，所以这里要前去1
        for i in range(len(num_neurons) - 1):
            self.params['W' + str(i + 1)] = np.random.randn(num_neurons[i], num_neurons[i+1]) * weight_scale
            self.params['b' + str(i + 1)] = np.zeros(num_neurons[i+1])
            # 这里处理的总循环式L-1，i最大为L-2，而batchnormalization只在层与层中间，也就是比如三个结点就只有两个间隔，所以这里是到L-2
            if self.normalization=='batchnorm' and i < len(num_neurons) - 2:
                self.params['beta' + str(i + 1)] = np.zeros([num_neurons[i+1]])
                self.params['gamma' + str(i + 1)] = np.ones([num_neurons[i+1]])
        ############################################################################
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
```

**目标：**

计算全连接网络的损失与梯度

**输入输出：**

```python
输入:
	- X: Array of input data of shape (N, d_1, ..., d_k)
	- y: Array of labels, of shape (N,). y[i] gives the label for X[i].

返回:
	If y is None, then run a test-time forward pass of the model and return:
	- scores: Array of shape (N, C) giving classification scores, where scores[i, c] is the classification score for X[i] and class c.
```

**完整实现：**

这里的实现思路就是按照上面一开始的注释提到的：

```python
{affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax
```

下面一起来看：具体代码在两行长#号中间：

下面依此调用affine、batch、relu、dropout的前向传播来实现！紧接着求出loss，最后来调用跟前向传播相对的反向传播来求梯度！

```python
def loss(self, X, y=None):
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        cache = {}
        scores = X 
        ############################################################################
        # 前向传播
        # {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax
        for i in range(1, self.num_layers + 1):
            scores, cache['fc'+str(i)] = affine_forward(scores, self.params['W' + str(i)], self.params['b' + str(i)])
            # Do not add relu, batchnorm, dropout after the last layer
            if i < self.num_layers: 
                if self.normalization == "batchnorm":
                    D = scores.shape[1]
                    # self.bn_params[i-1] since the provided code above initilizes bn_params for layers from index 0, here we index layer from 1. 
                    scores, cache['bn'+str(i)] = batchnorm_forward(scores, self.params['gamma'+str(i)], self.params['beta'+str(i)], self.bn_params[i-1]) 
                scores, cache['relu'+str(i)] = relu_forward(scores)
                if self.use_dropout:
                    scores, cache['dropout'+str(i)] = dropout_forward(scores, self.dropout_param)
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # 计算得分
        loss, last_grad = softmax_loss(scores, y)
        loss += 0.5 * self.reg * sum([np.sum(self.params['W' + str(i)]**2) for i in range(1, self.num_layers + 1)])
        ############################################################################
        
        ############################################################################
        # 反向传播
        for i in range(self.num_layers, 0, -1): 
            if i < self.num_layers: # No ReLU, dropout, Batchnorm for the last layer
                if self.use_dropout:
                    last_grad = dropout_backward(last_grad, cache['dropout' + str(i)])
                last_grad = relu_backward(last_grad, cache['relu' + str(i)])
                if self.normalization == "batchnorm":
                    last_grad, grads['gamma'+str(i)], grads['beta'+str(i)] = batchnorm_backward(last_grad, cache['bn'+str(i)])
            last_grad, grads['W' + str(i)], grads['b' + str(i)] = affine_backward(last_grad, cache['fc' + str(i)])
            grads['W' + str(i)] += self.reg * self.params['W' + str(i)]
        ############################################################################
        return loss, grads
```

## 4.训练模型

最后，回到FullyConnectedNets.ipynb文件中，依此调用即可，最后填充相应的训练一个好的模型的代码！

```python
hidden_dims = [100] * 4
range_weight_scale = [1e-2, 2e-2, 5e-3]
range_lr = [1e-5, 5e-4, 1e-5]

best_val_acc = -1
best_weight_scale = 0
best_lr = 0

print("Training...")

for weight_scale in range_weight_scale:
    for lr in range_lr:
        model = FullyConnectedNet(hidden_dims=hidden_dims, reg=0.0,
                                 weight_scale=weight_scale)
        solver = Solver(model, data, update_rule='adam',
                        optim_config={'learning_rate': lr},
                        batch_size=100, num_epochs=5,
                        verbose=False)
        solver.train()
        val_acc = solver.best_val_acc  
        print('Weight_scale: %f, lr: %f, val_acc: %f' % (weight_scale, lr, val_acc))
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weight_scale = weight_scale
            best_lr = lr
            best_model = model
print("Best val_acc: %f" % best_val_acc)
print("Best weight_scale: %f" % best_weight_scale)
print("Best lr: %f" % best_lr)
```

最终要求的精度在验证集上至少50%！

上面训练后的最好结果为：

```python
Validation set accuracy:  0.528
Test set accuracy:  0.527
```

