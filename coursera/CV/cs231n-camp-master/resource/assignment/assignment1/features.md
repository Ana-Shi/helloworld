

# Cs231n之features及numpy

## 0.说在前面

今天发现cs231n还差一个features.py未更新，特更，并且更新中间穿插的numpy使用！

有关更多内容，请关注微信公众号：guangcity

![](https://github.com/Light-City/images/blob/master/wechat.jpg?raw=true)

## 1.numpy

> 取指定位置的精度

**In**

```python
np.around([-0.6,1.2798,2.357,9.67,13], decimals=0)
```

**Out**

看到没，负数进位取绝对值大的！

```python
array([-1.,  1.,  2., 10., 13.])
```

**In**

```python
np.around([1.2798,2.357,9.67,13], decimals=1)
```

**Out**

```python
array([ 1.3,  2.4,  9.7, 13. ])
```

**In**

```python
np.around([1.2798,2.357,9.67,13], decimals=2)
```

**Out**

```python
array([ 1.28,  2.36,  9.67, 13.  ])
```

从上面可以看出，decimals表示指定保留有效数的位数，当超过5就会进位(此时包含5)！

但是，如果这个参数设置为负数，又表示什么？

**In**

```python
np.around([1,2,5,6,56], decimals=-1)
```

**Out**

```python
array([ 0,  0,  0, 10, 60])
```

发现没，当超过5时候(不包含5)，才会进位！-1表示看一位数进位即可，那么如果改为-2呢，那就得看两位！

例如：

**In**

```python
np.around([1,2,5,50,56,190], decimals=-2)
```

**Out**

```python
array([  0,   0,   0,   0, 100, 200])
```

看到没，必须看两位，超过50才会进位，190的话，就看后面两位，后两位90超过50，进位，那么为200！

> 计算沿指定轴第N维的离散差值

**In**

```python
x = np.arange(1 , 16).reshape((3 , 5))
```

**Out**

```python
array([[ 1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10],
       [11, 12, 13, 14, 15]])
```

**In**

```python
np.diff(x,axis=1) #默认axis=1
```

**Out**

```python
array([[1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1]])
```

**In**

```python
np.diff(x,axis=0) 
```

**Out**

```python
array([[5, 5, 5, 5, 5],
       [5, 5, 5, 5, 5]])
```

> 取整

**In**

```python
np.floor([-0.6,-1.4,-0.1,-1.8,0,1.4,1.7])
```

**Out**

```python
array([-1., -2., -1., -2.,  0.,  1.,  1.])
```

看到没，负数取整，跟上述的around一样，是向左！

> 取上限

```python
np.ceil([1.2,1.5,1.8,2.1,2.0,-0.5,-0.6,-0.3])
```

取上限！找这个小数的最大整数即可！

> 查找

利用`np.where`实现小于0的值用0填充吗，大于0的数不变！

**In**

```python
x = np.array([[1, 0],
       [2, -2],
     [-2, 1]])
```

**Out**

```python
array([[ 1,  0],
       [ 2, -2],
       [-2,  1]])
```

**In**

```python
np.where(x>0,x,0)
```

**Out**

```python
array([[1, 0],
       [2, 0],
       [0, 1]])
```

## 3.features

> svm

```python
iters = 6000
svm = LinearSVM()
for i in learning_rates:
    for j in regularization_strengths:
        svm.train(X_train_feats,y_train,learning_rate=i,reg=j,num_iters=iters)
        y_train_pred = svm.predict(X_train_feats)
        accu_train = np.mean(y_train == y_train_pred)
        y_val_pred = svm.predict(X_val_feats)
        accu_val = np.mean(y_val == y_val_pred)
        results[(i, j)] = (accu_train, accu_val)
        
        if best_val < accu_val:
            best_val = accu_val
            best_svm = svm
```

> neural network

```python
results = {}
best_val = -1
best_net = None

learning_rates = [1e-2 ,1e-1, 5e-1, 1, 5]
regularization_strengths = [1e-3, 5e-3, 1e-2, 1e-1, 0.5, 1]

for lr in learning_rates:
    for reg in regularization_strengths:
        net = TwoLayerNet(input_dim, hidden_dim, num_classes)
        # Train the network
        stats = net.train(X_train_feats, y_train, X_val_feats, y_val,
        num_iters=1500, batch_size=200,
        learning_rate=lr, learning_rate_decay=0.95,
        reg= reg, verbose=False)
        val_acc = (net.predict(X_val_feats) == y_val).mean()
        if val_acc > best_val:
            best_val = val_acc
            best_net = net         
        results[(lr,reg)] = val_acc
```


对比上述两者方法，发现代码差不多，其实就是调参！！！


















