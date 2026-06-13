# -*- coding: utf-8 -*-
from sklearn.cluster import KMeans
from matplotlib import cm
from sklearn.metrics import silhouette_samples
import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([3, 1, 1, 2, 1, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8])
x2 = np.array([5, 4, 5, 6, 5, 8, 6, 7, 6, 7, 1, 2, 1, 2, 3, 2, 3])

x = np.array(list(zip(x1, x2))).reshape(len(x1), 2)

km = KMeans(n_clusters=3,init="k-means++",n_init=10,max_iter=300,tol=1e-4,random_state=0)
y_km = km.fit_predict(x)

#获取簇的标号
cluster_labels = np.unique(y_km)
#获取簇的个数
n_clusters = cluster_labels.shape[0]
#基于欧式距离计算轮廓系数
silhouette_vals = silhouette_samples(x,y_km,metric="euclidean")
#设置y坐标的起始位置
y_ax_lower,y_ax_upper=0,0
yticks=[]
for i,c in enumerate(cluster_labels):
    #获取不同簇的轮廓系数
    c_silhouette_vals = silhouette_vals[y_km == c]
    #对簇中样本的轮廓系数由小到大进行排序
    c_silhouette_vals.sort()
    #获取到簇中轮廓系数的个数
    y_ax_upper += len(c_silhouette_vals)
    #获取不同颜色
    color = cm.jet(i / n_clusters)
    #绘制水平直方图
    plt.barh(range(y_ax_lower,y_ax_upper),c_silhouette_vals,height=1.0,edgecolor="none",color=color)
    #获取显示y轴刻度的位置
    yticks.append((y_ax_lower+y_ax_upper) / 2)
    #下一个y轴的起点位置
    y_ax_lower += len(c_silhouette_vals)
#获取轮廓系数的平均值
silhouette_avg = np.mean(silhouette_vals)
#绘制一条平行y轴的轮廓系数平均值的虚线
plt.axvline(silhouette_avg,color="g",linestyle="-")
#设置y轴显示的刻度
plt.yticks(yticks,cluster_labels+1)

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.ylabel("簇")
plt.xlabel("轮廓系数")
plt.show()