from math import log
import operator
import pickle
import chapter_three.treePlotter as mtp

# 计算信息熵
def calShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = labelCounts[key]/numEntries
        shannonEnt -= prob*log(prob, 2)
    return shannonEnt

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

# 待划分的数据集，划分数据集的特征，需要返回的特征值
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

# 获取最佳的划分特征
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0])-1
    baseEntropy = calShannonEnt(dataSet) # 计算labels的信息熵
    bestInfoGain = 0.0
    bestFeature = -1
    num = len(dataSet)
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/num # 取值为value的概率
            newEntropy += prob*calShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

# 计算最大分类
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 使用ID3算法构建决策树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 所有数据都属于某一类，停止递归
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 没有特征了
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

# 分类测试
def classify(inputTree, feaLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = feaLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], feaLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

# 保存决策树，相当于保存训练好的模型
def storeTree(inputTree, filename):
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()

# 获取保存的决策树
def grabTree(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)

def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head':{0:'no', 1:'yes'}}, 1: 'no'}}}}]
    return listOfTrees[i]

# 获取隐形眼睛数据
def getLensesData():
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    return lenses, lensesLabels

if __name__ == '__main__':
    # -----------建树并绘制树---------
    # myDat, labels = createDataSet()
    # myTree = createTree(myDat, labels)
    # mtp.createPlot(myTree)

    # ---测试算法，使用决策树执行分类---
    # 这里的labels是特征的名称
    # myDat, labels = createDataSet()
    # myTree = retrieveTree(0)
    # print(myTree)
    # result1 = classify(myTree, labels, [1, 0])
    # print(result1)
    # result2 = classify(myTree, labels, [1, 1])
    # print(result2)
    # filename = 'classifierStorage.txt'
    # storeTree(myTree, filename)
    # print(grabTree(filename))

    # ------使用决策树预测隐形眼镜类型--------
    lenses, lensesLabels = getLensesData()
    lensesTree = createTree(lenses, lensesLabels)
    print(lensesTree)
    mtp.createPlot(lensesTree)

    # bestFeaIndex = chooseBestFeatureToSplit(myDat)
    # print(bestFeaIndex)
    # result1 = splitDataSet(myDat, 0, 1)
    # print(result1)
    # result2 = splitDataSet(myDat, 0, 0)
    # print(result2)
    # print(myDat)
    # entropy = calShannonEnt(myDat)
    # 计算信息熵
    # print(entropy)

