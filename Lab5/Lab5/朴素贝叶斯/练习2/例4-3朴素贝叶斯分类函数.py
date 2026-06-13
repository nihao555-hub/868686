# -*- coding: utf-8 -*-
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
def trainNB0(trainMatrix,trainCategory):  #创建朴素贝叶斯分类器函数
  numTrainDocs=len(trainMatrix)
  numWords=len(trainMatrix[0])
  pAbusive=sum(trainCategory)/float(numTrainDocs)
  p0Num=np.ones(numWords);p1Num=np.ones(numWords)
  p0Deom=2.0;p1Deom=2.0
  for i in range(numTrainDocs):
    if trainCategory[i]==1:
      p1Num+=trainMatrix[i]
      p1Deom+=sum(trainMatrix[i])
    else:
      p0Num+=trainMatrix[i]
      p0Deom+=sum(trainMatrix[i])
  p1vect=np.log(p1Num/p1Deom)  #change to log
  p0vect=np.log(p0Num/p0Deom)  #change to log
  return p0vect,p1vect,pAbusive
def  classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
  p1=sum(vec2Classify*p1Vec)+np.log(pClass1)
  p0=sum(vec2Classify*p0Vec)+np.log(1.0-pClass1)
  if p1>p0:
    return 1
  else:
    return 0
def testingNB():
  listOPosts,listClasses=loadDataSet()
  myVocabList=createVocabList(listOPosts)
  trainMat=[]
  for postinDoc in listOPosts:
    trainMat.append(setOfWordseVec(myVocabList, postinDoc))
  p0V,p1V,pAb=trainNB0(np.array(trainMat),np.array(listClasses))
  print("p0V={0}".format(p0V))
  print("p1V={0}".format(p1V))
  print("pAb={0}".format(pAb))
  testEntry=['happy','freedom']
  thisDoc=np.array(setOfWordseVec(myVocabList, testEntry))
  print(thisDoc)
  print("vec2Classify*p0Vec={0}".format(thisDoc*p0V))
  print(testEntry,'classified as :',classifyNB(thisDoc, p0V, p1V, pAb))
  testEntry=['stupid']
  thisDoc=np.array(setOfWordseVec(myVocabList, testEntry))
  print(thisDoc)
  print(testEntry,'classified as :',classifyNB(thisDoc, p0V, p1V, pAb))
if __name__=='__main__':
  testingNB()
