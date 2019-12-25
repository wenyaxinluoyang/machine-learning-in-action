from math import log

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
    pass



if __name__ == '__main__':
    myDat, labels = createDataSet()
    result1 = splitDataSet(myDat, 0, 1)
    print(result1)
    result2 = splitDataSet(myDat, 0, 0)
    print(result2)
    # print(myDat)
    # entropy = calShannonEnt(myDat)
    # 计算信息熵
    # print(entropy)

