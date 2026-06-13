# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:36:57 2020

@author: Harry
"""

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]  #获取数据集的所有类别
    if classList.count(classList[0]) == len(classList): 
        return classList[0]							   #如果数据集的所有类别都相同则不需要划分，使用完所有特征后仍然不能将数据划分到某个类别上，则返回出现次数最多的类别
    if len(dataSet[0]) == 1: 
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)   	#获取数据集中按哪一列进行划分
    bestFeatLabel = labels[bestFeat]  				#bestFeatLabel=列描述
    myTree = {bestFeatLabel:{}} 					#创建一个字典
    del(labels[bestFeat]) 							#删除已计算过的列
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues) 					#获取某列所有不重复值
    for value in uniqueVals:
        subLabels = labels[:] 
        myTree[bestFeatLabel][value] = createTree(splitDataSet(
			dataSet, bestFeat, value),subLabels) 	#递归
    return myTree
myDat,labels = createDataSet()
myTree = createTree(myDat,labels)
print(myTree)
