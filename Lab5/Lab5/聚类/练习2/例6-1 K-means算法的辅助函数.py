# -*- coding: utf-8 -*-


from numpy import *
import numpy as np
mat = np.asmatrix

def loadDataSet(fileName):      
    '''
    从文件中加载数据矩阵
    param filename: 保存数据矩阵的文件名 str
    return dataSet: 数据矩阵   [[],[],...]  list(list)
    '''
    dataMat = []               
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine))
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA, vecB):
    '''
    计算两个向量的欧氏距离
    param vecA,vecB: 两个待计算距离的向量  numpy.ndarray
    return : 两个向量的欧氏距离
    '''
    return sqrt(sum(power(vecA - vecB, 2))) 

def randCent(dataSet, k):
    '''
    随机挑选k个初始簇中心
    param dataSet: 数据矩阵
    param k: 簇数
    return : k个初始簇中心
    '''
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j]) 
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids

datMat = mat(loadDataSet("testSet.txt"))
print(min(datMat[:,0]))
print(min(datMat[:,1]))
print(max(datMat[:,0]))
print(max(datMat[:,1]))
print(distEclud(datMat[0],datMat[1]))