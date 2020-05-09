from numpy import  *

import operator

def createDataSet():

    group=array([[1.2,1.3],[1.1,1.2],[1.4,1.3],[1.1,1.3],[3.0,3.0],[3.2,3.4],[3.0,3.2],[3.2,3.3]])
    labels=['A','A','A','A','B','B','B','B']
    return  group,labels

def createTestSet():

    group=array([[1.2,1.3],[1.1,1.2],[1.4,1.3],[1.1,1.3],[3.0,3.0],[3.2,3.4],[3.0,3.2],[3.2,3.3]])
    labels=['A','A','A','A','B','B','B','B']
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





















