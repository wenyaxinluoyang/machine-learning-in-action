import matplotlib.pyplot as plt
import chapter_three.trees as mts
# 判断节点
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
# 叶子节点
leafNode = dict(boxstyle="round4", fc="0.8")
# 箭头参数
arrow_args = dict(arrowstyle="<-")

# 获取叶子节点的个数
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict:
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else: numLeafs += 1
    return numLeafs
# 获取树的深度
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else: thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

# 绘制节点
def plotNode(nodeText, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeText, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)

# 绘制连线
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va='center', ha='center', rotation=30)
    # createPlot.ax1.text(xMid, yMid, txtString)

# 计算宽与高
def plotTree(myTree, parentPt, nodeText):
    numLeafs = getNumLeafs(myTree)
    # depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPr = (plotTree.xOff + (1.0 + numLeafs)/2/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPr, parentPt, nodeText)
    plotNode(firstStr, cntrPr, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPr, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPr, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPr, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = getNumLeafs(inTree)
    plotTree.totalD = getTreeDepth(inTree)
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()








if __name__ == '__main__':
    # createPlot()
    # print(retrieveTree(1))
    myTree = mts.retrieveTree(0)
    myTree['no surfacing'][3] = 'maybe'
    print(myTree)
    createPlot(myTree)
    # leafNum = getNumLeafs(myTree)
    # treeDepth = getTreeDepth(myTree)
    # print(leafNum, treeDepth)

