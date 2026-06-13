# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:53:19 2020

@author: Harry
"""

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      	#最后一个元素是当前实例的类别标签。
    baseEntropy = calcShannonEnt(dataSet)	#计算原始信息熵。
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        	#遍历数据集中所有特征。
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        uniqueVals = set(featList)       	#创建唯一的分类标签列表。
        newEntropy = 0.0
        for value in uniqueVals:     		#遍历当前特征中所有唯一的特征值。
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)  #计算每种划分方式的信息熵。   
        infoGain = baseEntropy - newEntropy    	#计算信息增益。
        if (infoGain > bestInfoGain):			#将结果与目前所得到的最优划分进行比较。
            bestInfoGain = infoGain  	 			#如果结果优于当前最优化分特征，则更新划分特征。
            bestFeature = i
    return bestFeature                     			#返回最优划分的索引值。
myDat,labels=createDataSet()
print(chooseBestFeatureToSplit(myDat))
print(myDat)
