import numpy as np
import matplotlib.pyplot as plt


def rhs(y,t,beta,nu):
    eq1=-beta*y[0]*y[1]
    eq2=beta*y[0]*y[1]-nu*y[1]
    #eq3=nu*y[1]
    # eq1=sp.Eq(sp.diff(S(t),t),-beta*S(t)*I(t))
    # eq2=sp.Eq(sp.diff(I(t),t),beta*S(t)*I(t)-nu*I(t))
    # eq3=sp.Eq(sp.diff(R(t),t),nu*I(t))
    return [eq1,eq2]

def initializePopulation(populationSize,gridSize):
    positions=np.random.randint(gridSize,size=(2,populationSize))
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

def diseaseSpreading(infected,susceptibles,recovered,transmissionRate,recoveryRate):
    numberOfInfected=len(infected[0])
    for k in range(numberOfInfected):
        #susceptibles[:][0] == infected[0][i]) & (susceptibles[:][1] == infected[1][i])
        indices = [i for i, x in enumerate(susceptibles[:][0]) if x == infected[0][k]]
        newlyInfected = [i for i in indices if susceptibles[1][i]==infected[1][k]]
        #reversing the list to get the proper elements when using pop
        newlyInfected.reverse()
        if newlyInfected != []:
            r = np.random.rand()
            if r < transmissionRate:
                for i in newlyInfected:
                    infected[0].append(susceptibles[0].pop(i))
                    infected[1].append(susceptibles[1].pop(i))

    #For popping recovered
    for k in reversed(range(numberOfInfected)):
        r = np.random.rand()
        if r<recoveryRate:
            recovered[0].append(infected[0].pop(k))
            recovered[1].append(infected[1].pop(k))

    return (infected,susceptibles,recovered)

if __name__=="__main__":
    recoveryRate=0.005
    populationSize=1000
    gridSize=100
    transmissionRate = 0.2
    diffusionRate=0.3
    numberOfInfected=10
    population = initializePopulation(populationSize, gridSize)
    infected=population[:numberOfInfected,:numberOfInfected].tolist()
    susceptibles=population[:,numberOfInfected:].tolist()
    recovered=[[],[]]

    #(infected, susceptibles, recovered)=diseaseSpreading(infected, susceptibles,recovered,transmissionRate,recoveryRate)
    plt.ion()

    fig,ax1=plt.subplots()
    ax1.set_ylim(0, gridSize)
    ax1.set_xlim(0, gridSize)
    h1,=ax1.plot(susceptibles[0], susceptibles[1], '*b')
    h2,=ax1.plot(infected[0],infected[1],'*r')
    h3,=ax1.plot(recovered[0],recovered[1],'*g')

    time=[0]
    numberOfSusceptible = [len(susceptibles[0])]
    numberOfInfected=[len(infected[0])]
    numberOfRecovered=[len(recovered[0])]

    fig2, ax2 = plt.subplots()
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title(r'$\beta =$' +str(transmissionRate)+'$,\ \gamma = 0.005$ , populationSize = 1000')
    ax2.set_ylim(0, 1000)
    ax2.set_xlim(0, 10000)
    h4,= ax2.plot(time,numberOfSusceptible,'b-',label='Susceptible')
    h5,= ax2.plot(time, numberOfInfected,'r-',label='Infected')
    h6,= ax2.plot(time, numberOfRecovered, 'g-',label='Recovered')
    plt.legend(handles=[h4,h5,h6], loc=1)
    plt.draw()
    while len(infected[0])>0:
        (infected, susceptibles, recovered) = diseaseSpreading(infected, susceptibles, recovered, transmissionRate,
                                                               recoveryRate)
        infected= updatePositions(infected, diffusionRate, gridSize)
        susceptibles = updatePositions(susceptibles, diffusionRate, gridSize)
        recovered = updatePositions(recovered, diffusionRate, gridSize)

        time.append(time[-1] + 1)
        numberOfSusceptible.append(len(susceptibles[0]))
        numberOfInfected.append(len(infected[0]))
        numberOfRecovered.append(len(recovered[0]))
        if time[-1]%100==0:
            h1.set_xdata(susceptibles[0])
            h1.set_ydata(susceptibles[1])
            h2.set_xdata(infected[0])
            h2.set_ydata(infected[1])
            h3.set_xdata(recovered[0])
            h3.set_ydata(recovered[1])

            h4.set_xdata(time)
            h4.set_ydata(numberOfSusceptible)
            h5.set_xdata(time)
            h5.set_ydata(numberOfInfected)
            h6.set_xdata(time)
            h6.set_ydata(numberOfRecovered)

            #writer.grab_frame()

            plt.draw()
            plt.pause(0.1)
    h1.set_xdata(susceptibles[0])
    h1.set_ydata(susceptibles[1])
    h2.set_xdata(infected[0])
    h2.set_ydata(infected[1])
    h3.set_xdata(recovered[0])
    h3.set_ydata(recovered[1])

    h4.set_xdata(time)
    h4.set_ydata(numberOfSusceptible)
    h5.set_xdata(time)
    h5.set_ydata(numberOfInfected)
    h6.set_xdata(time)
    h6.set_ydata(numberOfRecovered)
    ax2.set_xlim(0, time[-1])
    fig.savefig('Agents'+str(populationSize)+'.png')
    fig2.savefig('DiseaseSpreading'+str(populationSize)+'.png')