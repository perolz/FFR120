import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors
import os
import shutil


def growForest(gridSize, forestGrid, p):
    tmp = np.random.rand(gridSize, gridSize)
    newTrees = np.where(tmp < p, 1, 0)
    newForest = np.where(newTrees + forestGrid >= 1, 1, 0)

    return newForest


def updateModel(gridSize, forestGrid, p, f):
    newForest = growForest(gridSize, forestGrid, p)

    # for i in range(gridSize):
    #     for j in range(gridSize):
    #
    #         if forestGrid<1:
    #             r=np.random.rand()
    #             if r<p:
    #                 tmpGrid[i][j]=1
    fire = []
    r = np.random.rand()
    if f > r:
        fireStartingPoint = np.random.randint(0, gridSize - 1, 2)
        if (newForest[fireStartingPoint[0]][fireStartingPoint[1]] == 1):
            fire = fireOutbreak(newForest, gridSize, fireStartingPoint)
            newForest = newForest

    return newForest, fire


def fireOutbreak(newForest, gridSize, fireStartingPoint):
    fire = np.zeros([gridSize, gridSize])
    fire[fireStartingPoint[0]][fireStartingPoint[1]] = 1
    nodes = [fireStartingPoint]
    visitedNodes = np.zeros([gridSize, gridSize])
    visitedNodes[fireStartingPoint[0]][fireStartingPoint[1]] = 1
    while len(nodes) > 0:
        tmpNodes = []
        for i in nodes:
            neighbourNode = neighbours(i[0], i[1])
            for j in neighbourNode:
                if (visitedNodes[j[0]][j[1]] == 0 and newForest[j[0]][j[1]] == 1):
                    fire[j[0]][j[1]] = 1
                    visitedNodes[j[0]][j[1]] = 1
                    tmpNodes.append(j)
                elif (visitedNodes[j[0]][j[1]] == 0):
                    visitedNodes[j[0]][j[1]] = 1

        # plt.scatter(np.array(nodes).T[0],np.array(nodes).T[1])
        # plt.show()
        nodes = tmpNodes
    return fire


def neighbours(x, y):
    tmp = np.array([(x - 1, y), (x, y - 1), (x + 1, y), (x, y + 1)])
    tmp[tmp == 128] = 0
    tmp[tmp == -1] = 127

    return tmp

def setup():
    p = 0.001
    f = 0.3
    gridSize = 128
    forestGrid = np.zeros([gridSize, gridSize])
    return gridSize,forestGrid,p,f


def homeWork1():
    directory = r'C:\Users\Fudge\Documents\Programming\FFR120\Home Problem 2\Images'
    print("the dir is: %s" % os.listdir(os.getcwd()))
    if os.path.exists(directory):
        shutil.rmtree('Images')
    if not os.path.exists(directory):
        os.makedirs(directory)
    gridSize,forestGrid,p,f=setup()

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(1, 1, 1)
    plt.suptitle('Sumulation with f=%1.2f and p=%1.3f' % (f, p))
    ts = np.arange(gridSize)
    X, Y = np.meshgrid(ts, ts)
    cmap1 = colors.ListedColormap(['white', 'green'])
    bounds = [0, 0.5, 1.5, 2]
    norm1 = colors.BoundaryNorm(bounds, cmap1.N)

    cmap2 = colors.ListedColormap(['white', 'green', 'red'])
    bounds = [0, 1, 2]
    norm2 = colors.BoundaryNorm(bounds, cmap2.N)
    i = 1
    while (True):
        forestGrid, fire = updateModel(gridSize, forestGrid, p, f)
        if len(fire) == 0:
            img = ax.imshow(forestGrid, interpolation='nearest', origin='lower',
                            cmap=cmap1)
            plt.savefig('Images/%d.png' % i)
        else:
            fireSize = np.sum(np.sum(fire, axis=0))
            forestDensity = np.sum(np.sum(forestGrid, axis=0))
            img = ax.imshow(forestGrid + fire, interpolation='nearest', origin='lower',
                            cmap=cmap2)
            plt.savefig('Images/%d.png' % i)
            tmp = (2 > forestGrid+fire) & (forestGrid > 0)
            forestGrid = np.where(tmp, 1, 0)
        i = i + 1

def homeWork2():
    gridSize,forestGrid,p,f=setup()

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(1, 1, 1)
    plt.suptitle('Sumulation with f=%1.2f and p=%1.3f' % (f, p))
    i = 1
    while (True):
        forestGrid, fire = updateModel(gridSize, forestGrid, p, f)
        if len(fire) >0:
            fireSize = np.sum(np.sum(fire, axis=0))
            forestDensity = np.sum(np.sum(forestGrid, axis=0))
            newForest= np.zeros([gridSize, gridSize])
            while(np.sum(np.sum(newForest, axis=0))<forestDensity):
                newForest=growForest(gridSize, newForest, p)

            #forstätt härifrån blir roligt och lätt!
            iterator=0
            while iterator==0:
                fireStartingPoint=np.random.randint(0,128,size=(1,2))
                iterator=newForest[fireStartingPoint[0][0]][fireStartingPoint[0][1]]
            fireStartingPoint=fireStartingPoint[0]

            newFire=fireOutbreak(newForest, gridSize, fireStartingPoint)

            sizeOfFire=np.sum(np.sum(newFire,axis=0))
            tmpBoealean = (2 > forestGrid+fire) & (forestGrid > 0)

            forestGrid = np.where(tmpBoealean, 1, 0)
        i = i + 1

if __name__ == "__main__":
    #homeWork1()
    homeWork2()