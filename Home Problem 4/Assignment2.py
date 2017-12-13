import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import collections
from scipy import linalg
import pickle

def PerferentialGrowth(m, n0, p):
    edges = np.zeros([n0, n0])

    for i in range(n0):
        for j in range(i + 1):
            r = np.random.rand()
            if r < p:
                edges[i][j] = 1
                edges[j][i] = 1
    np.fill_diagonal(edges, 0)
    rows, cols = np.where(edges == 1)

    nodes = set([n1 for n1 in rows] + [n2 for n2 in cols])
    gr = nx.Graph()
    for node in nodes:
        gr.add_node(node)
    for i in range(len(rows)):
        gr.add_edge(rows[i], cols[i])

    for i in range(4000):
        alreadyConnected = []
        degree_sequence = sorted([d for n, d in gr.degree()], reverse=True)
        numberOfNodes = len(gr.nodes)
        gr.add_node(numberOfNodes)
        totalDegree = sum(degree_sequence)
        while len(alreadyConnected) < m:
            # Ändra här. Ska inte vara random utan bero på fördelningen sedan innan
            # Använd degree
            iterator = 0
            comparisonValue = degree_sequence[0] / totalDegree
            r1 = np.random.rand()
            while r1 > comparisonValue:
                iterator = iterator + 1
                comparisonValue = comparisonValue + degree_sequence[iterator] / totalDegree

            if not iterator in alreadyConnected:
                gr.add_edge(numberOfNodes, iterator)
                alreadyConnected.append(iterator)
    return gr

if __name__ == '__main__':
    m=5
    n0=200
    #gr=PerferentialGrowth(m,n0,0.3)
    gr=pickle.load(open("graph.pickle",'rb'))
    pos = nx.spring_layout(gr)

    #pickle._dump(gr,open( "graph.pickle", "wb" ))
    fig, ax = plt.subplots()
    nx.draw(gr, pos, node_size=10, cmap='jet', with_label=True)
    plt.show()