# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 19:05:25 2020

@author: Harry
"""
import numpy as np
from numpy import *
import operator
def fileMatrix(filename):
    file = open(filename)
    arrayOLines = file.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        print(listFromLine)
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals

def classify(inX,dataSet,labels,k):
    # 计算距离
    dataSetSize = dataSet.shape[0] #读取矩阵第一行的长度 4
    diffMat = tile(inX,(dataSetSize,1)) - dataSet #tile 重复inx的各个维度1次    相减
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    # 选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    # 排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def classifyPerson():
    #输出结果
    resultList = ['讨厌', '有些喜欢', '非常喜欢']
    #三维特征用户输入
    precentTats = float(input("玩视频游戏所耗时间百分比:"))
    ffMiles = float(input("每年获得的飞行常客里程数:"))
    iceCream = float(input("每周消费的冰激淋公升数:"))
    #打开的文件名
    filename = "datingTestSet2.txt"
    #打开并处理数据
    datingDataMat, datingLabels = fileMatrix(filename)
    #训练集归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #生成NumPy数组,测试集
    inArr = np.array([ffMiles, precentTats, iceCream])
    #测试集归一化
    norminArr = (inArr - minVals) / ranges
    #返回分类结果
    classifierResult = classify(norminArr, normMat, datingLabels, 3)
    #打印结果
    print("你可能%s这个人"%(resultList[classifierResult - 1]))
if __name__ == '__main__':
    classifyPerson()
