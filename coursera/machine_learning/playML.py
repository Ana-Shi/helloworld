
# coding: utf-8

# In[5]:


import numpy as np

class SimpleLinearRegression1:
    def __init__(self):
        """初始化SimpleLinearRegression模型"""
        self.a_ = None
        self.b_ = None
        
    def fit(self,x_train,y_train):
        """根据训练数据集x_train,y_train训练Simple Linear Regression"""
        assert x_train.ndim == 1,            "SimpleLinearRegression can only sove one feature"
        assert x_train.ndim == y_train.ndim,            "the size of x_train must be equal to the size of y_train"
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        num = 0.0
        d=0.0
        for x_i,y_i in zip(x_train,y_train):
            num += (x_i-x_mean)*(y_i-y_mean)
            d += (x_i-x_mean)**2
        self.a_ = num/d
        self.b_ = y_mean - self.a_*x_mean
        return self
    def predict(self,x_predict):
        """给定待预测数据集x_predict,返回表示x_predict的结果向量"""
        assert x_predict.ndim ==1,        "SimpleLinearRegression can only sove one feature"
        assert self.a_ is not None and self.b_ is not None,            "must fit before predict"
        return np.array([self._predict(x) for x in x_predict])
    
    def _predict(self,x_single):
        """给定单个待预测数据x_single,返回x_single的预测结果"""
        return self.a_*x_single + self.b_
    
    def __repr__(self):
        return 'SimpleLinearRegression1'


# In[ ]:


class SimpleLinearRegression2:
    def __init__(self):
        """初始化SimpleLinearRegression模型"""
        self.a_ = None
        self.b_ = None
        
    def fit(self,x_train,y_train):
        """根据训练数据集x_train,y_train训练Simple Linear Regression"""
        assert x_train.ndim == 1,            "SimpleLinearRegression can only sove one feature"
        assert x_train.ndim == y_train.ndim,            "the size of x_train must be equal to the size of y_train"
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        num = (x_train-x_mean).dot(y_train-y_mean)
        d = (x_train-x_mean).dot(x_train-x_mean)
        d=0.0
        for x_i,y_i in zip(x_train,y_train):
            num += (x_i-x_mean)*(y_i-y_mean)
            d += (x_i-x_mean)**2
        self.a_ = num/d
        self.b_ = y_mean - a*x_mean
        return self
    def predict(self,x_predict):
        """给定待预测数据集x_predict,返回表示x_predict的结果向量"""
        assert x_predict.ndim ==1,        "SimpleLinearRegression can only sove one feature"
        assert self.a_ is not None and self.b_ is not None,            "must fit before predict"
        return np.array([self._predict(x) for x in x_predict])
    
    def _predict(self,x_single):
        """给定单个待预测数据x_single,返回x_single的预测结果"""
        return self.a_*x_single + self.b_
    
    def __repr__(self):
        return 'SimpleLinearRegression2'

