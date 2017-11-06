import sympy as sp
import numpy as np
import scipy
import matplotlib.pyplot as plt


def rhs(y,t,beta,nu):
    eq1=-beta*y[0]*y[1]
    eq2=beta*y[0]*y[1]-nu*y[1]
    #eq3=nu*y[1]
    # eq1=sp.Eq(sp.diff(S(t),t),-beta*S(t)*I(t))
    # eq2=sp.Eq(sp.diff(I(t),t),beta*S(t)*I(t)-nu*I(t))
    # eq3=sp.Eq(sp.diff(R(t),t),nu*I(t))
    return [eq1,eq2]

def initializePopulation(populationSize,gridSize,numberOfInfected):
    positions=np.random.randint(gridSize,size=(2,populationSize))
    # carryingDisease=np.concatenate((np.zeros(populationSize-numberOfInfected),np.ones(numberOfInfected)))
    # tmp=np.vstack([positions,carryingDisease])
    return positions

def updatePositions(population,diffusionRate,gridSize):
    for i in range(len(population[0])):
        r=np.random.rand()
        if r<diffusionRate:
            #Choose direction not outside grid
            possibleDirections=np.arange(4)
            if(population[0][i]==gridSize):
                possibleDirections=possibleDirections[possibleDirections!=0]
            if(population[0][i]==0):
                possibleDirections=possibleDirections[possibleDirections != 1]
            if(population[1][i]==gridSize):
                possibleDirections=possibleDirections[possibleDirections != 2]
            if(population[1][i]==0):
                possibleDirections=possibleDirections[possibleDirections != 3]

            direction=np.random.choice(possibleDirections)
            if(direction==0):
                population[0][i] =population[0][i]+1
            elif(direction==1):
                population[0][i] = population[0][i] - 1
            elif (direction == 2):
                population[1][i] = population[1][i] + 1
            elif (direction == 3):
                population[1][i] = population[1][i] - 1
    return population

def diseaseSpreading(infected,susceptibles,transmissionRate):
    test=len(infected[0])
    for i in range(len(infected[0])):

        newlyInfected=np.where((susceptibles[:][0]==infected[0][i]) & (susceptibles[:][1]==infected[1][i]))
        print(newlyInfected[0])
        if newlyInfected[0]!=[]:
            for j in range(2):
                infected[j]=np.concatenate(infected[j],)
            print(infected)


            r=np.random.rand()
            if r<transmissionRate:
                np.append(infected,susceptibles[0][newlyInfected],)


if __name__=="__main__":
    recoveryRate=1/5
    populationSize=1000
    gridSize=10
    transmissionRate = 0.7/populationSize
    diffusionRate=0.4
    numberOfInfected=5
    population = initializePopulation(populationSize, gridSize,numberOfInfected)
    #population[:numberOfInfected,:numberOfInfected]
    infected=np.array(2,populationSize-numberOfInfected)
    susceptibles=population[:,numberOfInfected:]
    recovered=np.zeros(2,populationSize)

    diseaseSpreading(infected, susceptibles,transmissionRate)
    plt.ion()
    fig,ax1=plt.subplots()
    ax1.set_ylim(0, gridSize)
    ax1.set_xlim(0, gridSize)
    h1,=ax1.plot(susceptibles[0], susceptibles[1], '*')
    h2,=ax1.plot(infected[0],infected[1],'*r')
    h3,=ax1.plot(recovered[0],recovered[1],'*r')
    plt.draw()
    for i in range(1000):
        diseaseSpreading(infected, susceptibles,transmissionRate)

        infected= updatePositions(infected, diffusionRate, gridSize)
        susceptibles = updatePositions(susceptibles, diffusionRate, gridSize)
        h1.set_xdata(susceptibles[0])
        h1.set_ydata(susceptibles[1])
        h2.set_xdata(infected[0])
        h2.set_ydata(infected[1])
        plt.draw()
        plt.pause(0.005)
