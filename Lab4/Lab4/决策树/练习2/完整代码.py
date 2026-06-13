# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 22:02:03 2020

@author: Harry
"""

from math import log
import operator
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)               	#声明数据集中样本总数
    labelCounts = {}                         	#创建字典
    for featVec in dataSet:                  	#所有可能分类的数量和发生频率
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) #log base 2
    return shannonEnt
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels
myDat,labels=createDataSet()
print(myDat)
print(calcShannonEnt(myDat))

def splitDataSet(dataSet, axis, value):
    retDataSet = []                               	#创建列表对象引用数据集，防止由于多次调用而改变元数据集。
    for featVec in dataSet: 						#遍历数据集中的每个元素
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  		
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)	#将符合特征的数据抽取出来
    return retDataSet
myDat,labels=createDataSet()
print(myDat)
print(splitDataSet(myDat,0,1))
print(splitDataSet(myDat,0,0))

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
# 返回出现次数最多的类别
def majorityCnt(classList):			
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]  # 获取数据集的所有类别
    if classList.count(classList[0]) == len(classList): 
        return classList[0]							   # 如果数据集的所有类别都相同则不需要划分，使用完所有特征后仍然不能将数据划分到某个类别上，则返回出现次数最多的类别
    if len(dataSet[0]) == 1: 
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)   	# 获取数据集中按哪一列进行划分
    bestFeatLabel = labels[bestFeat]  				# bestFeatLabel=列描述
    myTree = {bestFeatLabel:{}} 					# 创建一个字典
    del(labels[bestFeat]) 							# 删除已计算过的列
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues) 					# 获取某列所有不重复值
    for value in uniqueVals:
        subLabels = labels[:] 
        myTree[bestFeatLabel][value] = createTree(splitDataSet(
			dataSet, bestFeat, value),subLabels) 	# 递归
    return myTree
myDat,labels = createDataSet()
myTree = createTree(myDat,labels)
print(myTree)
