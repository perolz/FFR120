import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors
import pickle
import networkx as nx
import collections
import scipy, scipy.stats,scipy.sparse

def show_graph_with_labels(adjacency_matrix,p):
    rows,cols = np.where(adjacency_matrix == 1)

    nodes = set([n1 for n1 in rows] + [n2 for n2 in cols])
    tmp=[]
    gr = nx.Graph()
    for node in nodes:
        gr.add_node(node)
    for i in range(len(rows)):
        gr.add_edge(rows[i],cols[i])
    pos = nx.shell_layout(gr)
    fig, ax = plt.subplots()


    nodes = gr.nodes()
    degree = gr.degree()
    edges= gr.edges()
    print(edges)
    color = [degree[n] for n in nodes]
    degree_sequence = sorted([d for n, d in gr.degree()], reverse=True)
    nx.draw(gr, pos, node_size=10, node_color=color, cmap='jet', with_label=True)

    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    numberOfNodes=len(adjacency_matrix)

    # plt.bar(deg, cnt, width=0.80, color='b')
    #
    # plt.title("Degree Histogram")
    # plt.ylabel("Count")
    # plt.xlabel("Degree")
    # tmp=deg[0::4]
    # ax.set_xticks([d + 0.4 for d in tmp])
    # ax.set_xticklabels(tmp)

    return fig,ax,gr


def ErdosRenyi(n,p):

    edges = np.zeros([n, n])
    #edges= scipy.sparse.rand(n,n,density=p).ceil()

    for i in range(n):
        for j in range(i+1):
            r=np.random.rand()
            if r<p:
                edges[i][j]=1
                edges[j][i]=1
    np.fill_diagonal(edges, 0)
    #edges = np.where(tmp < p, 1, 0)
    fig, ax= show_graph_with_labels(edges)

    x = np.arange(scipy.stats.binom.ppf(0.0001, numberOfNodes, 0.01),
                  scipy.stats.binom.ppf(0.99999, numberOfNodes, 0.01))
    ax.plot(x, scipy.stats.binom.pmf(x, numberOfNodes, 0.01) * numberOfNodes, 'r', ms=8)
    fig.savefig('ErdosRenyi.png')
    plt.show()

def SmallWorld(n,p,c):
    edges = np.zeros([n, n])
    for i in range(c):
        np.fill_diagonal(edges[i+1:],1)
        np.fill_diagonal(edges[:,i+1:],1)
        np.fill_diagonal(edges[-(i + 1):], 1)
        np.fill_diagonal(edges[:, -(i + 1):], 1)
    fig,ax,gr = show_graph_with_labels(edges,p)

    nodes=gr.nodes()
    for i in edges:
        r = np.random.rand()
        if r<p:
            r1=np.random.randint(0,len(nodes))
            r2=np.random.randint(0,len(nodes))
            if not((r1,r2) in edges):
                gr.add_edge(r1, r2)

    pos = nx.shell_layout(gr)
    fig, ax = plt.subplots()
    nx.draw(gr, pos, node_size=10, node_color='blue', cmap='jet', with_label=True)
    plt.show()

def PerferentialGrowth(m,n0,p):
    edges = np.zeros([n0, n0])

    for i in range(n0):
        for j in range(i+1):
            r=np.random.rand()
            if r<p:
                edges[i][j]=1
                edges[j][i]=1
    np.fill_diagonal(edges, 0)
    rows, cols = np.where(edges == 1)

    nodes = set([n1 for n1 in rows] + [n2 for n2 in cols])
    gr = nx.Graph()
    for node in nodes:
        gr.add_node(node)
    for i in range(len(rows)):
        gr.add_edge(rows[i], cols[i])

    for i in range(100):
        alreadyConnected = []
        numberOfNodes=len(gr.nodes)
        gr.add_node(numberOfNodes)
        while len(alreadyConnected)<m:
            #Ändra här. Ska inte vara random utan bero på fördelningen sedan innan
            #Använd degree
            r1=np.random.randint(0,numberOfNodes-1)
            if not r1 in alreadyConnected:
                gr.add_edge(numberOfNodes,r1)
                alreadyConnected.append(r1)

    print('hej')


if __name__ == "__main__":
    # Method for excercise 1
    numberOfNodes=200
    p=0.1
    c=2
    #ErdosRenyi(8000,0.01)
    # show_graph_with_labels(adjacency)
    #SmallWorld(numberOfNodes,p,c)
    PerferentialGrowth(10,30,0.2)

