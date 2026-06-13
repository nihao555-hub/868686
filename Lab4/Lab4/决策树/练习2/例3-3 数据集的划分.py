# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:49:23 2020

@author: Harry
"""

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
