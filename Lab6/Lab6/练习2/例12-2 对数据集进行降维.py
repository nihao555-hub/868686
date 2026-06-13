# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
def demean(data):
    #  axis=0 是 矩阵data减矩阵data每一列的均值（即axis=0
    return data - np.mean(data, axis=0)

# pca 目标函数推导公式
def f(w, data):
    return np.sum((data.dot(w)**2)) / len(data)

# 求目标函数对应的梯度
def df_math(w, data):
    #即通过数学推导得出的求梯度公式
    return data.T.dot(data.dot(w)) * 2. / len(data)

def direction(w):
    # 由于w不一定是一个单位向量，把它转化成一个单位向量， w/w.模 
    return w / np.linalg.norm(w)

def gradient_ascent(df, data, initial_w, eta, n_iters = 1e4, epsilon=1e-8):
    
    w = direction(initial_w) 
    cur_iter = 0

    while cur_iter < n_iters:
        gradient = df(w, data)
        last_w = w
        w = w + eta * gradient
        w = direction(w) # 注意1：每次求一个单位方向，方便能得到一个合适的值方便计算
        if(abs(f(w, data) - f(last_w, data)) < epsilon):
            break
            
        cur_iter += 1

    return w
data = np.empty((200, 2))
data[:,0] = np.random.uniform(0., 100., size=200)
data[:,1] = 0.75 * data[:,0] + 1.5 + np.random.normal(0, 10., size=200)

data_mean = demean(data)  #去均值处理
initial_w = np.random.random(data.shape[1]) # 注意：不能用0向量开始
eta = 0.001 #  初始化步长
w = gradient_ascent(df_math, data_mean, initial_w, eta)  #降维并得出第一个方向，即第一个主成分，使得在该方向上进行映射可以有最大的间距（方差）

plt.scatter(data_mean[:,0], data_mean[:,1])
plt.plot([0, w[0]*30], [0, w[1]*30], color='r') # 将w表示的方向进行绘制，为了更清楚地展示，这里适当扩大w的表示范围（这里扩大了30倍），但是w0和w1比例要保持一致。
plt.show()