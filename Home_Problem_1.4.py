from Home_Problem_1 import *
from mpl_toolkits.mplot3d import Axes3D

if __name__=="__main__":
    # Denna ska skrivas om till att ha många olika värden. Det är så jag kommer kunna få rätt data.
    recoveryRate=0.005
    populationSize=1000
    gridSize=100
    transmissionRate = 0.1

    diffusionRate=0.3
    initialNumberOfInfected=10
    #transmissionRateSettings = []
    correspondingRecovered = []
    #initialSpread=[]
    numberOfIndexes=15
    transmissionRateSettings=np.linspace(0.1,0.5,numberOfIndexes)
    initialSpread=np.linspace(10,150,numberOfIndexes)
    xv,yv=np.meshgrid(initialSpread,transmissionRateSettings)
    correspondingRecovered=np.empty([numberOfIndexes,numberOfIndexes])
    for k in range(len(xv)):
        for i in range(len(yv)):
            population = initializePopulation(populationSize, gridSize)
            print(xv[k][i])
            infected=population[:int(xv[k][i]),:int(xv[k][i])].tolist()
            susceptibles=population[:,int(xv[k][i]):].tolist()
            recovered=[[],[]]

            averageRecovered = []

            for j in range(10):
                while len(infected[0])>0:
                    (infected, susceptibles, recovered) = diseaseSpreading(infected, susceptibles, recovered, yv[k][i],
                                                                           recoveryRate)
                    infected= updatePositions(infected, diffusionRate, gridSize)
                    susceptibles = updatePositions(susceptibles, diffusionRate, gridSize)
                    recovered = updatePositions(recovered, diffusionRate, gridSize)
                averageRecovered.append(len(recovered[0]))
            correspondingRecovered[k][i]=(sum(averageRecovered)/len(averageRecovered))


    fig =plt.figure(figsize=[16,9])

    ax = fig.add_subplot(111, projection='3d')
    #xv,yv=np.meshgrid(transmissionRateSettings,initialSpread)
    #zv=np.meshgrid(correspondingRecovered,correspondingRecovered)
    #plt.scatter(transmissionRateSettings,initialSpread,correspondingRecovered)
    ax.plot_surface(xv,yv, correspondingRecovered)
    #plt.contour(transmissionRateSettings,initialSpread,correspondingRecovered,antialiased=True)
    # plt.plot([0.05,0.5],[Rinf,Rinf],'r')
    # plt.text(0.01,Rinf,r'$R_\infty$ = %1.1f' % Rinf)

    ax.set_xlabel('initialSpread',fontsize=14)
    ax.set_ylabel('transmissionRateSettings',fontsize=14)
    ax.set_zlabel('correspondingRecovered',fontsize=14)
    # plt.title(r'Test with different values of recovery rates $\beta$=%1.1f' %transmissionRate)
    #plt.legend(handles=[h1], loc=1)
    fig.savefig('Plotfor1.4_Beta%1.1f.png' %transmissionRate)
    plt.show()
    with open('1.4Data.txt','w') as f:
        f.write(str(xv) + '\n\n' +str(yv) + '\n\n' +str(correspondingRecovered) + '\n\n' )
