# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:45:22 2020

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
