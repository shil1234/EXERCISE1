import numpy as  np

from  math  import  log

#计算香浓熵
def  CalcShannonEnt(DataSet):
    num=len(DataSet)
    labelCounts={}
    for  featVec  in  DataSet:
        currentLabel=featVec[-1]
        if  currentLabel not in  labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    shannonEnt=0
    for i  in  labelCounts:
        p=labelCounts[i]/num
        shannonEnt-=p*log(p,2)
    return  shannonEnt


def  createDataSet():
    dataSet=[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,0,'no'],[0,1,'no']]
    labels=['no surfacing','flippers']
    return dataSet,labels

dataSet,Labels=createDataSet()
shannonEnt=CalcShannonEnt(dataSet)
print(shannonEnt)

#给定特征划分数据集


def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.extend(reducedFeatVec)
    return retDataSet








