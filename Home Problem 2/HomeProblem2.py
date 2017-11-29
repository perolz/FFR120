import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors
import os
import shutil
import pickle
from scipy.optimize import curve_fit
from scipy import interpolate

def growForest(gridSize, forestGrid, p):
    tmp = np.random.rand(gridSize, gridSize)
    newTrees = np.where(tmp < p, 1, 0)
    newForest = np.where(newTrees + forestGrid >= 1, 1, 0)

    return newForest


def updateModel(gridSize, forestGrid, p, f):
    newForest = growForest(gridSize, forestGrid, p)

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
            neighbourNode = neighbours(i[0], i[1],gridSize)
            for j in neighbourNode:
                if (visitedNodes[j[0]][j[1]] == 0 and newForest[j[0]][j[1]] == 1):
                    fire[j[0]][j[1]] = 1
                    visitedNodes[j[0]][j[1]] = 1
                    tmpNodes.append(j)
                elif (visitedNodes[j[0]][j[1]] == 0):
                    visitedNodes[j[0]][j[1]] = 1

        nodes = tmpNodes
    return fire


def neighbours(x, y,gridSize):
    tmp = np.array([(x - 1, y), (x, y - 1), (x + 1, y), (x, y + 1)])
    tmp[tmp == gridSize] = 0
    tmp[tmp == -1] = gridSize-1

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


    i = 1
    fireSize=[]
    fireSizeReference=[]
    while (i<100000):
        forestGrid, fire = updateModel(gridSize, forestGrid, p, f)
        if len(fire) > 0:
            fireSize.append(np.sum(np.sum(fire, axis=0)))
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

            fireSizeReference.append( np.sum(np.sum(newFire,axis=0)))

            tmpBoealean = (2 > forestGrid+fire) & (forestGrid > 0)

            forestGrid = np.where(tmpBoealean, 1, 0)
        i = i + 1

    fireSize=np.array(fireSize)
    fireSize=-np.sort(-fireSize)/128**2
    fireSizeReference=np.array(fireSizeReference)
    fireSizeReference=-np.sort(-fireSizeReference)/(128**2)
    tspace=np.arange(1,len(fireSize)+1)/len(fireSize)
    print(len(fireSize))
    print(len(tspace))
    with open("tmp.pickle", "wb") as f:
        pickle.dump((fireSize, fireSizeReference,tspace), f)

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Sumulation with f=%1.2f and p=%1.3f' % (f, p))
    plt.xlabel('Relative fire size')
    plt.ylabel('cCDF')
    plt.loglog(fireSize,tspace,'b',label='Fire normal')
    plt.loglog(fireSizeReference,tspace,'r',label='Reference if there was no fire before')
    plt.show()


def reprintData():
    with open('task4.pickle','rb') as f:
         exponents= pickle.load(f)

    # fig = plt.figure(figsize=(10, 10))
    #
    # ax = fig.add_subplot(1, 1, 1)
    # ax.set_title('Sumulation with f=%1.2f and p=%1.3f' % (0.3, 0.001))
    # plt.xlabel('Relative fire size')
    # plt.ylabel('cCDF')
    #
    # line1, = plt.loglog(fireSize, tspace, 'b', label='Fire normal')
    # line2, = plt.loglog(fireSizeReference, tspace, 'r', label='Same forest density, but no fire before')
    # plt.legend(handles=[line1, line2])
    # plt.savefig('Images_to_show/Taskb.png')
    # plt.show()

    gridSize=np.array([8,16,32,64,128,256,512])
    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Plot with different lattices and extrapolation')
    plt.xlabel('1/N')
    plt.ylabel(r'$\tau$')

    x = np.linspace(0, 0.13, 50)
    f = interpolate.interp1d(1 / gridSize, exponents, fill_value='extrapolate')

    tmp = f(0)
    ax.text(0.04, 1.3, 'Value for when N goes to infinity=%1.3f' % tmp)
    line1,=plt.plot(x, f(x),label='Extrapolation of values')
    line2,=plt.plot(1 / gridSize, exponents,'bo',label='Values')
    plt.legend(handles=[line1, line2])
    plt.savefig('Images_to_show/Taskd.png')
    plt.show()

def func(x,a,b):
    return x**a*10**-b

def homeWork3():
    with open('tmp.pickle','rb') as f:
        (fireSize, fireSizeReference, tspace) = pickle.load(f)

    tau = 1.3
    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(r'Fitting of data with $\tau$=%1.2f' % (tau))
    plt.xlabel('Relative fire size')
    plt.ylabel('cCDF')
    line1, = plt.loglog(fireSize, tspace, 'b', label='Fire distribution')

    line2,=plt.loglog(tspace[:int(len(tspace)/16)],func(tspace[:int(len(tspace)/16)],1-tau,1.21),'red',label='Data fitting')
    popt, pcov = curve_fit(func, fireSize, tspace)
    print(popt)
    plt.legend(handles=[line1,line2])
    plt.savefig('Images_to_show/Taskc_lines.png')

    X=np.zeros(len(fireSize))
    xmin=1
    for i in range(len(fireSize)):
        r=np.random.random()
        X[i]=xmin*(1-r)**(-1/(tau-1))
    X= -np.sort(-X) / (128 ** 2)
    fig2 = plt.figure(figsize=(10, 10))

    ax2 = fig.add_subplot(1, 1, 1)
    ax2.set_title(r'Fitting of data with $\tau$=%1.2f' % (tau))
    plt.xlabel('Relative fire size')
    plt.ylabel('cCDF')
    line1, = plt.loglog(fireSize, tspace, 'b', label='Fire distribution')
    line2, = plt.loglog(X[round(len(X)/16):],tspace[round(len(X)/16):],'r',label='Synthetic power law data')

    plt.legend(handles=[line1, line2])
    plt.savefig('Images_to_show/Taskc_powerlaw_only_fitted.png')
    plt.show()

def homeWork4():
    gridSize=np.array([8,16,32,64,128,256,512])
    exponents=[]
    p=0.001
    f=0.3
    for j in gridSize:

        forestGrid = np.zeros([j, j])
        i = 1
        fireSize = []
        while (i < 100000):
            forestGrid, fire = updateModel(j, forestGrid, p, f)
            if len(fire) > 0:
                fireSize.append(np.sum(np.sum(fire, axis=0)))
                tmpBoealean = (2 > forestGrid + fire) & (forestGrid > 0)
                forestGrid = np.where(tmpBoealean, 1, 0)
            i = i + 1
        fireSize = np.array(fireSize)
        fireSize = -np.sort(-fireSize) / j ** 2
        tspace = np.arange(1, len(fireSize) + 1) / len(fireSize)

        popt, pcov = curve_fit(func, fireSize, tspace)
        exponents.append(1-popt[0])
    with open('task4.pickle','wb') as f:
        pickle.dump(exponents,f)
    exponents=np.array(exponents)

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(1, 1, 1)
    plt.xlabel('1/N')
    plt.ylabel(r'$\tau$')
    plt.plot(1/gridSize,exponents)
    x = np.linspace(0, 1, 50)
    f = interpolate.interp1d(1 / gridSize, exponents, fill_value='extrapolate')

    plt.plot(x,f(x))


    ax.axis([0, 10, 0, 10])
    plt.savefig('Images_to_show/Taskd.png')
    plt.show()


if __name__ == "__main__":
    #homeWork1()
    #homeWork2()
    reprintData()
    #homeWork3()
    #homeWork4()