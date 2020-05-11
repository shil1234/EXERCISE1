from numpy import  *
import random
import operator
import matplotlib
import matplotlib.pyplot as plt


def createDataSet():


    group = array([[1, 2], [1.1, 2], [1.12, 1.3], [0.9, 1.3], [2.9, 3.0], [2.2, 3.4], [4, 3.2], [3.2, 2.8],[4.9, 4.0], [4.2, 4.4], [4, 4.2], [4.2, 4.8]])
    labels=['A','A','A','A','B','B','B','B','C','C','C','C']
    return  group,labels

def createTestSet():
    group = array([[1.2, 1.3], [1.1, 1.2], [1.4, 1.3], [1.1, 1.3], [3.0, 3.0], [3.2, 3.4], [3.0, 3.2], [3.2, 3.3], [2.8, 3.1], [3.3, 3.3]])
    labels=['A','A','A','A','B','B','B','B','B','C','C']
    return  group,labels

def classify(inx,dataset,labels,k):
    dataSetSize=dataset.shape[0]
    diffMat=tile(inx,[dataSetSize,1])-dataset
    sqDiffMat=diffMat**2
    sqDistance=sqDiffMat.sum(axis=1)
    distances=sqDistance**0.5
    sortedDissIndicies=distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDissIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),
                            reverse=True)
    return sortedClassCount[0][0]

testSingel=array([2, 2])

dataset,labels=createDataSet()
result=classify(testSingel,dataset,labels,4)

print(result)

#测试算法：作为完整程序验证分类器
def datingClassTest():
    hoRatio = 1  #设置测试集比重，前10%作为测试集，后90%作为训练集
    datingDataMat,datingLabels = createTestSet()
    m = datingDataMat.shape[0]    #得到样本数量m
    numTestVecs = int(m*hoRatio)    #得到测试集最后一个样本的位置
    errorCount = 0.0    #初始化定义错误个数为0
    for i in range(m):
        #测试集中元素逐一放进分类器测试，k = 3
        classifierResult = classify(datingDataMat[i],dataset,labels,3)
        #输出分类结果与实际label
        # print("the classifier came back with: %d, the real answer is: %d"% (classifierResult, datingLabels[i]))
        #若预测结果与实际label不同，则errorCount+1
        if (classifierResult !=datingLabels[i]): errorCount += 1.0
        #输出错误率 = 错误的个数 / 总样本个数
        print(numTestVecs)
        print("the total error rate is: %f" % (errorCount/float(numTestVecs)))

datingClassTest()

#------------------------------------------------------------------------------------------
#约会网站配对
import  numpy as np

#准备数据

def file2matrix(filename):
    fr=open(filename)
    arrayOLines=fr.readlines()
    numberofLines=len(arrayOLines)
    returnMat=zeros((numberofLines, 3))
    classLabelVector=[]
    indes=0
    for  line in  arrayOLines:
        line =line.strip()
        listFromLine=line.split('\t')
        returnMat[indes,:]=listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        indes=indes+1

    return returnMat,classLabelVector

dataMat,dataLabels=file2matrix('datingTestSet2.txt')

print(dataMat[0:20,:])

print(dataLabels[0:20])

# import matplotlib
#
# import matplotlib.pyplot as plt
#
# fig=plt.figure()
#
# ax=fig.add_subplot(111)
#
# ax.scatter(dataMat[:,0],dataMat[:,1],15*array(dataLabels),15*array(dataLabels))
#
# plt.show()

#归一化数据
def autoNorm(dataSet):
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=dataSet-tile(minVals,(m,1))
    normDataSet=normDataSet/tile(maxVals,(m,1))
    return normDataSet,ranges,minVals


a,b,c=autoNorm(dataMat)

print(a)
print(b)
print(c)

#构建完整的系统


















