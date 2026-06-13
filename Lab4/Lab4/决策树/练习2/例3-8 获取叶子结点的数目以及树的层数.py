# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:38:31 2020

@author: Harry
"""

def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0] 		#字典的第一个键，即树的一个结点
    secondDict = myTree[firstStr]  			#这个键的值，对应该结点的所有分支
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs +=1
    return numLeafs
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth
