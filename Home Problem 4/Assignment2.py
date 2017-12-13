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

def SmallWorld(n, p, c):
    adjacency_matrix = np.zeros([n, n])
    for i in range(c):
        np.fill_diagonal(adjacency_matrix[i + 1:], 1)
        np.fill_diagonal(adjacency_matrix[:, i + 1:], 1)
        np.fill_diagonal(adjacency_matrix[-(i + 1):], 1)
        np.fill_diagonal(adjacency_matrix[:, -(i + 1):], 1)
    #fig, ax, gr = show_graph_with_labels(edges, p)
    rows, cols = np.where(adjacency_matrix == 1)

    nodes = set([n1 for n1 in rows] + [n2 for n2 in cols])
    tmp = []
    gr = nx.Graph()
    for node in nodes:
        gr.add_node(node)
    for i in range(len(rows)):
        gr.add_edge(rows[i], cols[i])

    nodes = gr.nodes()
    edges= gr.edges()
    for i in edges:
        r = np.random.rand()
        if r < p:
            r1 = np.random.randint(0, len(nodes))
            r2 = np.random.randint(0, len(nodes))
            if not ((r1, r2) in edges):
                tmp.append((r1,r2))
    gr.add_edges_from(tmp)
    return gr

def phi(R):
    tmp=1 / (1 + 1 / R)
    return tmp
if __name__ == '__main__':
    m=5
    n0=200
    #gr=PerferentialGrowth(m,n0,0.3)
    networksize=1000
    gr=nx.fast_gnp_random_graph(networksize,0.01)
    #gr=SmallWorld(4000,0.05,4)
    #gr=pickle.load(open("graph.pickle",'rb'))
    #pos = nx.spring_layout(gr)
    edges=gr.edges()
    R=0.5
    edgesToRemove=[]
    Rvalues=np.linspace(0.001,0.6,1000)
    largestCluster = []
    for j in Rvalues:
        average=[]
        for k in range(10):
            grtmp = gr.copy()
            tmpEdges = grtmp.edges()
            edgesToRemove=[]
            for i in edges:
                r=np.random.random()
                if(r<1-phi(j)):
                    edgesToRemove.append((i[0],i[1]))

            grtmp.remove_edges_from(edgesToRemove)
            for i in tmpEdges:
                grtmp.add_edge(i[1], i[0])
            giant = max(nx.connected_component_subgraphs(grtmp), key=len)
            average.append(len(giant.nodes())/networksize)

        largestCluster.append(sum(average)/len(average))
        #print(len(giant.nodes()))
        #pos = nx.spring_layout(giant)
        #nx.draw(giant, pos, node_size=10, cmap='jet', with_label=True)



    plt.plot(Rvalues,largestCluster)
    plt.xlabel('R')
    plt.ylabel('Relative size of largest cluster')
    print(largestCluster)
    plt.show()
    # giant = max(nx.connected_component_subgraphs(grtmp), key=len)
    # print(len(giant.nodes()))


    #pickle._dump(gr,open( "graph.pickle", "wb" ))
    #fig, ax = plt.subplots()




