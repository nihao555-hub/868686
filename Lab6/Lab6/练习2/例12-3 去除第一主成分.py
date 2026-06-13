# -*- coding: utf-8 -*-

def f(w, data):
    return np.sum((data.dot(w)**2)) / len(data)

def df(w, data):
    return data.T.dot(data.dot(w)) * 2. / len(data)

def direction(w):
    return w / np.linalg.norm(w)

def first_component(data, initial_w, eta, n_iters = 1e4, epsilon=1e-8):
    
    w = direction(initial_w) 
    cur_iter = 0

    while cur_iter < n_iters:
        gradient = df(w, data)
        last_w = w
        w = w + eta * gradient
        w = direction(w) 
        if(abs(f(w, data) - f(last_w, data)) < epsilon):
            break
            
        cur_iter += 1

    return w
data = demean(data)
initial_w = np.random.random(data.shape[1])    
eta = 0.01
w = first_component(data, initial_w, eta) #这里求解出第一个主成分
data_2 = np.empty(data.shape)  # 初始1个矩阵，和data的维度保持一致，用来存储去除掉第一个主成分分量后的样本
for i in range(len(data)):
    data_2[i] = data[i] - data[i].dot(w) * w  # 得到每个样本在去除掉第一个主成分分量后的值

#将该分量绘制出来
plt.scatter(data_2[:,0], data_2[:,1])
plt.show() 
w2 = first_component(data_2, initial_w, eta)

# 第一主成分和第二主成分结果相乘接近0，因为第一主成分和第二主成分垂直，所以矩阵相乘为0
w.dot(w2)