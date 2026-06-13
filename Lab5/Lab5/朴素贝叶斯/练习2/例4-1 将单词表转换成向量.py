# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:56:01 2019

@author: Harry
"""
import numpy as np
#创建实验样本
def loadDataSet():
	  #词条切分后的文档集合
  postingList=[['I', 'am', 'happy', 'to', 'join', 'with', 'you'],\
         ['today', 'in', 'what', 'will', 'go', 'down', 'in', 'stupid'],\
         ['history', 'as', 'the', 'greatest', 'demonstration', 'for', 'freedom', 'in'],\
         ['the', 'history', 'stupid', 'of', 'our'],\
         ['nation', 'In', 'a', 'sense', 'we', 'have', 'come', 'to', 'our'],\
         ['nation', 'capital', 'to', 'cash', 'a', 'stupid']]  #单词类别标签集合
  classVec = [0,1,0,1,0,1]  #1表示单词列表中出现侮辱性单词, 0表示没有出现侮辱性单词
  return postingList,classVec 
#创建一个包含文档中出现的所有不重复单词的列表
def createVocabList(dataset):
  vocabSet=set([])                     #创建一个空列表
  for document in dataset:
    vocabSet=vocabSet|set(document) #将每篇文档返回的新词集合添加到上面创建的空列表中。操作符“|”用于求两个集合的并集
  return list(vocabSet)
#处理文本，返回文档向量
def setOfWordseVec(vocabList,inputSet):
  returnVec=[0]*len(vocabList)             #返回一个与词汇表等长的全0向量
  for word in inputSet:
    if word in vocabList:
      returnVec[vocabList.index(word)]=1  #vocabList.index()函数获取vocabList列表某个元素的位置，这段代码得到一个只包含0和1的列表
    else:
      print("没有这个单词 :%s!"%word)
  return returnVec
listOPosts,listClasses=loadDataSet()
myVocabList=createVocabList(listOPosts)
print(len(myVocabList)) #返回单词列表的长度
print(myVocabList)       #返回单词列表中的单词
print(setOfWordseVec(myVocabList, listOPosts[0])) #返回第1条词条中的单词在myVocabList出现的位置
print(setOfWordseVec(myVocabList, listOPosts[5])) #返回第6条词条中的单词在myVocabList出现的位置
