import operator
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
from os import listdir

# 创建数据集和标签
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# k近邻分类
def classify0(intX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # 求xi-xj , tile 可以理解为简单复制n个输入向量，n为dataset样本数
    diffMat = tile(intX, (dataSetSize,1)) - dataSet
    # (xi-xj)^2
    sqDiffMat = diffMat**2
    # 计算距离
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 获取海伦的约会数据
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


# 归一化  (oldValue-min)/(max-min), 因为各种变量量纲不一样
def autoNorm(dataSet):
    # 求每列参数最小值，最大值
    minVals = dataSet.min(0) # 第一个参数是轴
    maxVals = dataSet.max(0)
    ranges = maxVals-minVals
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals

# 测试约会人分类器效果
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0] # 行数
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], 3)
        print(f'the classifierResult came back with {classifierResult}, the real answer is: {datingLabels[i]}')
        if classifierResult != datingLabels[i]: errorCount += 1
    print(f'the total error rate is: %f' % (errorCount/numTestVecs))

# 约会网站预测函数
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input('percentage of time spent playing video games?'))
    ffMiles = float(input('frequent flier miles earned per year?'))
    iceCream = float(input('liters of ice cream consumed per year?'))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print('You will probably like this person:', resultList[classifierResult-1])


# 把图像转化为向量
def img2vector(filename):
    returnVec = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVec[0,32*i+j] = int(lineStr[j])
    return returnVec

# 手写数字识别系统的测试代码
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('./digits/trainingDigits/')
    m = len(trainingFileList) # 训练样本个数
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('./digits/trainingDigits/' + fileNameStr)
    testFileList = listdir('./digits/testDigits/')
    errorCount = 0
    mTest = len(testFileList) # 测试样本的个数
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('./digits/testDigits/' + fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print(f'the classifier came back with: {classifierResult}, the real answer is: {classNumStr}')
        if classifierResult != classNumStr: errorCount += 1
    print('\n the total number of errors is:', errorCount)
    print('\n the total error rate is:', (errorCount/mTest))



if __name__ == '__main__':
    handwritingClassTest()
    # print(imgVec[0])
    # 算法测试
    # datingClassTest()
    # 算法使用
    # classifyPerson()
    # group, labels = createDataSet()
    # label = classify0([0,0], group, labels, 3)
    # print(label)
    # datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    # normDataSet, ranges, minVals = autoNorm(datingDataMat)
    # print(normDataSet)
    # print(datingDataMat)
    # print(datingLabels[0:20])
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:,1], datingDataMat[:, 2])
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0*array(datingLabels), 15.0*array(datingLabels))
    # ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0*array(datingLabels), 15.0*array(datingLabels))
    # plt.show()






