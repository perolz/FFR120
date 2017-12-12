import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import collections
import scipy, scipy.stats, scipy.sparse


def show_graph_with_labels(adjacency_matrix, p):
    rows, cols = np.where(adjacency_matrix == 1)

    nodes = set([n1 for n1 in rows] + [n2 for n2 in cols])
    tmp = []
    gr = nx.Graph()
    for node in nodes:
        gr.add_node(node)
    for i in range(len(rows)):
        gr.add_edge(rows[i], cols[i])
    pos = nx.shell_layout(gr)
    fig, ax = plt.subplots()

    nodes = gr.nodes()
    degree = gr.degree()
    edges = gr.edges()
    print(edges)
    color = [degree[n] for n in nodes]
    degree_sequence = sorted([d for n, d in gr.degree()], reverse=True)
    nx.draw(gr, pos, node_size=10, node_color=color, cmap='jet', with_label=True)

    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    numberOfNodes = len(adjacency_matrix)

    # plt.bar(deg, cnt, width=0.80, color='b')
    #
    # plt.title("Degree Histogram")
    # plt.ylabel("Count")
    # plt.xlabel("Degree")
    # tmp=deg[0::4]
    # ax.set_xticks([d + 0.4 for d in tmp])
    # ax.set_xticklabels(tmp)

    return fig, ax, gr


def ErdosRenyi(n, p):
    edges = np.zeros([n, n])
    # edges= scipy.sparse.rand(n,n,density=p).ceil()

    for i in range(n):
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
    pos = nx.spring_layout(gr)
    fig, ax = plt.subplots()

    nodes = gr.nodes()
    degree = gr.degree()
    color = [degree[n] for n in nodes]
    vmin = min(color)
    vmax = max(color)
    nodes = nx.draw_networkx_nodes(gr, pos, node_color=color, node_size=10, cmap='jet', with_labels=False)
    edges = nx.draw_networkx_edges(gr, pos)

    plt.colorbar(nodes)
    # #edges = np.where(tmp < p, 1, 0)
    # fig, ax= show_graph_with_labels(edges)
    #
    # x = np.arange(scipy.stats.binom.ppf(0.0001, numberOfNodes, 0.01),
    #               scipy.stats.binom.ppf(0.99999, numberOfNodes, 0.01))
    # ax.plot(x, scipy.stats.binom.pmf(x, numberOfNodes, 0.01) * numberOfNodes, 'r', ms=8)
    # fig.savefig('ErdosRenyi.png')
    fig.savefig('ErdosNetwork.png')
    plt.show()


def SmallWorld(n, p, c):
    edges = np.zeros([n, n])
    for i in range(c):
        np.fill_diagonal(edges[i + 1:], 1)
        np.fill_diagonal(edges[:, i + 1:], 1)
        np.fill_diagonal(edges[-(i + 1):], 1)
        np.fill_diagonal(edges[:, -(i + 1):], 1)
    fig, ax, gr = show_graph_with_labels(edges, p)

    nodes = gr.nodes()
    for i in edges:
        r = np.random.rand()
        if r < p:
            r1 = np.random.randint(0, len(nodes))
            r2 = np.random.randint(0, len(nodes))
            if not ((r1, r2) in edges):
                gr.add_edge(r1, r2)

    pos = nx.shell_layout(gr)
    fig, ax = plt.subplots()
    nx.draw(gr, pos, node_size=10, node_color='blue', cmap='jet', with_label=True)
    plt.show()


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
    nodes = gr.nodes()
    degree = gr.degree()
    color = [degree[n] for n in nodes]
    pos = nx.spring_layout(gr)
    fig, ax = plt.subplots()
    vmin = min(color)
    vmax = max(color)
    nodes = nx.draw_networkx_nodes(gr, pos, node_color=color, node_size=10, cmap='jet', with_labels=False)
    edges = nx.draw_networkx_edges(gr, pos)
    # nx.draw(gr, pos, node_size=10, nodelist=nodes,node_color=color, cmap='jet', with_label=True)

    plt.colorbar(nodes)
    fig.savefig('NetworkPlot_PerferentialGrowth_n0%d_m%d_1000extraNodes.png' % (n0, m))
    degree_sequence = sorted([d for n, d in gr.degree()], reverse=True)
    cumulative = [np.sum(degree_sequence[:i]) / np.sum(degree_sequence) for i in range(len(degree_sequence))]
    inverseDegree = 1 / np.array(cumulative)
    theoretical = lambda k: 2 * m ** 2 * k ** -2
    tspan = range(len(degree_sequence))
    yval = [theoretical(x) for x in tspan[1:]]
    fig2 = plt.figure(2)
    line1, = plt.loglog(tspan, inverseDegree, label='Experimental')
    line2, = plt.loglog(tspan[1:], yval, label='Theoretical')
    plt.legend(handles=[line1, line2])
    plt.ylabel('Number of connections')
    fig2.savefig('Loglog_PerferentialGrowth_n0%d_m%d_1000extraNodes.png' % (n0, m))
    plt.show()


if __name__ == "__main__":
    # Method for excercise 1
    numberOfNodes = 20
    p = 0.4

    c = 2
    # ErdosRenyi(8000,0.01)
    # show_graph_with_labels(adjacency)
    # SmallWorld(numberOfNodes,p,c)
    PerferentialGrowth(10, 50, 0.2)
