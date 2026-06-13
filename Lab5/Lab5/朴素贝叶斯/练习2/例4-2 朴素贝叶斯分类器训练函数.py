# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:57:54 2019

@author: Harry
"""
import numpy as np
def loadDataSet():

    
  postingList=[['I', 'am', 'happy', 'to', 'join', 'with', 'you'],\
         ['today', 'in', 'what', 'will', 'go', 'down', 'in', 'stupid'],\
         ['history', 'as', 'the', 'greatest', 'demonstration', 'for', 'freedom', 'in'],\
         ['the', 'history', 'stupid', 'of', 'our'],\
         ['nation', 'In', 'a', 'sense', 'we', 'have', 'come', 'to', 'our'],\
         ['nation', 'capital', 'to', 'cash', 'a', 'stupid']]
  classVec = [0,1,0,1,0,1]  #1 is abusive, 0 not
  return postingList,classVec
def createVocabList(dataset):
  vocabSet=set([])
  for document in dataset:
    vocabSet=vocabSet|set(document)
  return list(vocabSet)
def setOfWordseVec(vocabList,inputSet):
  returnVec=[0]*len(vocabList)
  for word in inputSet:
    if word in vocabList:
      returnVec[vocabList.index(word)]=1  #vocabList.index() 函数获取vocabList列表某个元素的位置，这段代码得到一个只包含0和1的列表
    else:
      print("没有这个单词 :%s !"%word)
  return returnVec
def trainNB0(trainMatrix,trainCategory):  #创建朴素贝叶斯分类器训练函数
  numTrainDocs=len(trainMatrix) #输入参数为文档矩阵
  numWords=len(trainMatrix[0])
  pAbusive=sum(trainCategory)/float(numTrainDocs) # trainCategory表示由每篇文档类别标签所构成的向量
  p0Num=np.ones(numWords);p1Num=np.ones(numWords) #初始化p0（属于非侮辱性词汇的概率）和p1（属于侮辱性词汇的概率），将所有词的出现数初始化为1
  p0Deom=2.0;p1Deom=2.0 #将分母初始化为2
  #遍历训练集trainMatrix中的所有文档
  for i in range(numTrainDocs):
    #每出现一次对应分类的单词都在该词的对应个数（p1Num或者p0Num）和文档的总词数中+1
    if trainCategory[i]==1:     
      p1Num+=trainMatrix[i]
      p1Deom+=sum(trainMatrix[i])
    else:
      p0Num+=trainMatrix[i]
      p0Deom+=sum(trainMatrix[i])
  #对每个元素除以该类别中的总词数
  p1vect=np.log(p1Num/p1Deom)  #转换成log形式
  p0vect=np.log(p0Num/p0Deom)  #转换成log形式
  return p0vect,p1vect,pAbusive
listOPosts,listClasses=loadDataSet()
myVocabList=createVocabList(listOPosts) #从预先加载值中调入数据
trainMat=[]                 #构建了一个包含所有词的列表myVocabList
#使用词向量来填充trainMat列表
for postinDoc in listOPosts:
  trainMat.append(setOfWordseVec(myVocabList, postinDoc))
p0V,p1V,pAb=trainNB0(trainMat, listClasses) #属于侮辱性文档的概率以及两个类别的概率向量
print("p0的概率log值为：")
print (p0V)
print("p1的概率log值为：")
print (p1V)
print("pAb的概率为：")
print (pAb)
