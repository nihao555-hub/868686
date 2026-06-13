# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:36:12 2020

@author: Harry
"""

#返回出现次数最多的类别
def majorityCnt(classList):			
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
