# -*- coding: utf-8 -*-


import re
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

