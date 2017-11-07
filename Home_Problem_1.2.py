from Home_Problem_1 import *

if __name__=="__main__":
    recoveryRate=0.005
    populationSize=1000
    gridSize=100
    transmissionRate = 0.2
    diffusionRate=0.3
    initialNumberOfInfected=200
    population = initializePopulation(populationSize, gridSize)
    infected=population[:initialNumberOfInfected,:initialNumberOfInfected].tolist()
    susceptibles=population[:,initialNumberOfInfected:].tolist()
    recovered=[[],[]]

    plt.ion()

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

    h4.set_xdata(time)
    h4.set_ydata(numberOfSusceptible)
    h5.set_xdata(time)
    h5.set_ydata(numberOfInfected)
    h6.set_xdata(time)
    h6.set_ydata(numberOfRecovered)
    ax2.set_xlim(0, time[-1])
    plt.draw()
    fig2.savefig('DiseaseSpreading_1.2_'+str(initialNumberOfInfected)+'.png')