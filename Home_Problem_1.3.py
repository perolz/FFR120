from Home_Problem_1 import *

if __name__=="__main__":
    # Denna ska skrivas om till att ha många olika värden. Det är så jag kommer kunna få rätt data.
    recoveryRate=0.005
    populationSize=1000
    gridSize=100
    transmissionRate = 0.4

    diffusionRate=0.3
    initialNumberOfInfected=10
    recoverySettingsRate = []
    correspondingRecovered = []

    for i in range(20):
        recoverySettingsRate.append(recoveryRate*(i+1))
        population = initializePopulation(populationSize, gridSize)
        infected=population[:initialNumberOfInfected,:initialNumberOfInfected].tolist()
        susceptibles=population[:,initialNumberOfInfected:].tolist()
        recovered=[[],[]]

        averageRecovered = []

        for j in range(20):
            while len(infected[0])>0:
                (infected, susceptibles, recovered) = diseaseSpreading(infected, susceptibles, recovered, transmissionRate,
                                                                       recoverySettingsRate[-1])
                infected= updatePositions(infected, diffusionRate, gridSize)
                susceptibles = updatePositions(susceptibles, diffusionRate, gridSize)
                recovered = updatePositions(recovered, diffusionRate, gridSize)
            averageRecovered.append(len(recovered[0]))
        correspondingRecovered.append(sum(averageRecovered)/len(averageRecovered))

    plotValues = [x / transmissionRate for x in recoverySettingsRate]

    fig =plt.figure(figsize=[16,9])
    h1, = plt.plot(plotValues, correspondingRecovered, 'g-', label='Recovered')

    Rinf=sum(correspondingRecovered[-5:])/5
    plt.plot([0.05,0.5],[Rinf,Rinf],'r')
    plt.text(0.01,Rinf,r'$R_\infty$ = %1.1f' % Rinf)

    plt.xlabel(r'$\beta/\gamma$',fontsize=14)
    plt.ylabel(r'$R_\infty$',fontsize=14)
    plt.title(r'Test with different values of recovery rates $\beta$=%1.1f' %transmissionRate)
    plt.legend(handles=[h1], loc=1)
    fig.savefig('Plotfor1.3_Beta%1.1f.png' %transmissionRate)
    plt.show()

