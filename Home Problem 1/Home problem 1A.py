import matplotlib.pyplot as plt
from Home_Problem_1 import initializePopulation
from Home_Problem_1 import updatePositions

if __name__=="__main__":
    populationSize=1
    gridSize=10
    diffusionRate=0.7
    population = initializePopulation(populationSize, gridSize)

    fig, ax1 = plt.subplots()
    ax1.set_ylim(0, gridSize)
    ax1.set_xlim(0, gridSize)
    plt.title('diffusionRate=0.7')
    h1, = ax1.plot(population[0], population[1], '*b')

    for i in range(100):
        population = updatePositions(population, diffusionRate, gridSize)
        h1.set_xdata(population[0])
        h1.set_ydata(population[1])
        plt.draw()
        plt.pause(0.05)