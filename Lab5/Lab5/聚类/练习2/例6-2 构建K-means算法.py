# -*- coding: utf-8 -*-
"""
Created on Sat May  9 11:09:46 2020

@author: Harry
"""
from numpy import *
import numpy as np
mat = np.asmatrix

def loadDataSet(fileName):      
#解析文件，按tab分割字段，得到一个浮点数字类型的矩阵
    dataMat = []        				 		#文件的最后一个字段是类别标签      
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine))		#将每个元素转成float类型
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA, vecB):					 	# 计算欧几里得距离
    return sqrt(sum(power(vecA - vecB, 2))) 	#返回两个向量之间的距离

def randCent(dataSet, k): 			#构建聚簇中心，取k个(此例中为4)随机质心
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n))) 	#每个质心有n个坐标值，总共要k个质心
    for j in range(n):
        minJ = min(dataSet[:,j]) 
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2))) #用于存放该样本属于哪类及质心距离 
    #clusterAssment第一列存放该数据所属的中心点，第二列是该数据到中心点的距离
    centroids = createCent(dataSet, k)
    clusterChanged = True  				#用于判断聚类是否已经收敛
    while clusterChanged:
        clusterChanged = False
        for i in range(m):				#把每一个数据点划分到离它最近的中心点
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j #如果第i个数据点到第j个中心点更近，则将i归属为j
            if clusterAssment[i,0] != minIndex: clusterChanged = True #如果分配发生变化，则继续迭代
            clusterAssment[i,:] = minIndex,minDist**2	#将第i个数据点的分配情况存入字典
        print(centroids)
        for cent in range(k): 			#重新计算中心点
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]] 
            centroids[cent,:] = mean(ptsInClust, axis=0) #计算这些数据的中心点 
    return centroids, clusterAssment

datMat = mat(loadDataSet("testSet.txt"))
myCentroids,clustAssing = kMeans(datMat,4)
print("质心坐标：", myCentroids)
print("分类结果：",clustAssing)