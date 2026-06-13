# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:39:36 2020

@author: Harry
"""
def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)  					#当前树的叶子数
    depth = getTreeDepth(myTree) 						#没有用到这个变量
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]
    #cntrPt是文本中心点，parentPt指向文本中心点 
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt) 				#画分支上的键
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD  	#从上往下画
    for key in secondDict.keys():
    #如果是字典则是一个判断（内部）结点
        if type(secondDict[key]).__name__=='dict': 
            plotTree(secondDict[key],cntrPt,str(key))       
        else:   #打印叶子结点
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])			       #定义横纵坐标轴  
    createPlot.ax1 = plt.subplot(111, frameon=False) 
    plotTree.totalW = float(getNumLeafs(inTree))   	#全局变量宽度 = 叶子数
    plotTree.totalD = float(getTreeDepth(inTree))  	#全局变量高度 = 深度
    #图形的大小是0-1 ，0-1
    plotTree.xOff = -0.5/plotTree.totalW;  
    #例如绘制3个叶子结点，坐标应为1/3,2/3,3/3
    #但这样会使整个图形偏右因此初始的，将x值向左移一点。
    plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()
if __name__ == '__main__':
    myTree = retrieveTree(0)
    createPlot(myTree)