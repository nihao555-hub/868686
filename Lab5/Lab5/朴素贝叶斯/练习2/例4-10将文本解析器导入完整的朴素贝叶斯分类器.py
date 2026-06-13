# -*- coding: utf-8 -*-

from numpy import *
import re
def createVocabList(dataSet):
    vocabSet = set([])  #创建空列表
    #因为传入是二维数组，所以将二维数组内的所有元素全部压入Set中（顺序可能会被打乱）
    #最后再以list的形式返回
    for document in dataSet:
        vocabSet = vocabSet | set(document) 
    return list(vocabSet)

#统计单词出现的次数，用于创建向量集
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)                     #测试数据集数目  6
    numWords = len(trainMatrix[0])                      #总单词（去重）数目  32
    pAbusive = sum(trainCategory)/float(numTrainDocs)   #该文档属于侮辱类的概率=被标记为侮辱类句子数量/总句子数量=3/6.0=0.5
    #变量初始化
    p0Num = zeros(numWords); p1Num = zeros(numWords)    #标记向量初始化为[0,0,0,0...]
    p0Denom = 0; p1Denom = 0                            #统计数为0

    #计算概率时，需要计算多个概率的乘积以获得文档属于某个类别的概率
    #即计算p(w0|ci)*p(w1|ci)*...p(wN|ci)，然后当其中任意一项的值为0，那么最后的乘积也为0.
    #为降低这种影响，采用拉普拉斯平滑，在分子上添加a(一般为1)，分母上添加ka(k表示类别总数)，
    #即在这里将所有词的出现数初始化为1，并将分母初始化为2*1=2
    p0Num = ones(numWords); p1Num = ones(numWords)        
    p0Denom = 2.0; p1Denom = 2.0                        

    #对于每个句子
    #如果该句被人工标记为侮辱性的，则其中出现的每个词汇p1Num都该被认为是侮辱性的，侮辱性词汇总数p1Denom也做相应统计
    #如果该句不是侮辱性的，同样做统计
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #每个单词的是侮辱词的条件概率=在侮辱词中出现的次数p1Num/侮辱词出现总数p1Denom
    p1Vect = p1Num/p1Denom
    p0Vect = p0Num/p0Denom
    #计算概率时，由于大部分因子都非常小，最后相乘的结果四舍五入为0,造成下溢出或者得不到准确的结果，
    #所以，我们可以对成绩取自然对数，即求解对数似然概率。这样，可以避免下溢出或者浮点数舍入导致的错误。
    #同时采用自然对数处理不会有任何损失。
    p1Vect = log(p1Num/p1Denom)          
    p0Vect = log(p0Num/p0Denom)          
    return p0Vect,p1Vect,pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    #p1=(单词A出现的次数*单词A出现在侮辱句时的概率+单词B出现的次数*单词B出现在侮辱句时的概率+...)*正常句出现的概率
    #p0=(单词A出现的次数*单词A出现在正常句时的概率+单词B出现的次数*单词B出现在正常句时的概率+...)*正常句出现的概率
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)   
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0
#文件解析及完整的垃圾邮件测试函数
def textParse(bigString):    #接收一个字符串并将其切分为字符串列表
    listOfTokens = re.split(r'\W', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] #去掉少于3个字符的字符串，并将所有字符串小写化
def spamTest():#对贝叶斯垃圾邮件分类器进行自动化处理
    docList=[]; classList = []; fullText =[]
    for i in range(1,26): #导入并解析文本文件 导入文件夹spam和ham下的文件文本，并将其解析为词列表
        #分别读取25封垃圾邮件和普通邮件
        wordList = textParse(open('邮件/垃圾邮件/%d.txt' % i).read())
        docList.append(wordList)  	#append表示追加
        fullText.extend(wordList) 	#extend表示扩展
        classList.append(1) 			#1代表垃圾邮件
        #因ham/23.txt中包含商标R符号，读取时需要忽略掉错误
        wordList = textParse(open('邮件/普通邮件/%d.txt' % i,encoding='utf-8',errors='ignore').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    #vocabulary 去重
    vocabList = createVocabList(docList)         #创建词汇表
    trainingSet = list(range(50)) 				#训练数据集：整数列表，值从0-49
    testSet=[]        							#测试数据集
    for i in range(10): #将整个交叉验证过程重复10次，求平均错误率
        #numpy包含ramdom，random.uniform用于生成一个0到len(trainingSet)的随机数
        randIndex = int(random.uniform(0,len(trainingSet))) #随机选取10个文件作为测试数据集
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  #将测试数据集的整数列表从训练数据集中删去

    trainMat=[]; trainClasses = []
    #剩下的40封邮件用于统计训练
    for docIndex in trainingSet:    
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex])) #构建词向量，训练矩阵
        trainClasses.append(classList[docIndex]) #训练数据集标签
    #计算每种条件对应的概率
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses)) #计算分类所需概率

    #选中的10封由于测试
    errorCount = 0
    for docIndex in testSet:        #对测试数据集分类
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        #如果用贝叶斯分类器的结果和实际结果不一样
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print ("classification error",docList[docIndex])
    #计算平均错误率
    print ('错位率为: ',float(errorCount)/len(testSet))
spamTest()