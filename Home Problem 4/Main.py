import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import collections
from scipy import linalg
import csv

def network():
    test = 0

def Kernighan_Lin(s,B,m):
    bestModelarity=Modelarity(s,B,m)
    while True:
        modelarities=[]
        for i in range(len(s)):
            tmpS=s
            tmpS[i]=-s[i]
            modelarities.append(Modelarity(tmpS,B,m))
        bestIndex=np.argmax(modelarities)
        if modelarities[bestIndex]>bestModelarity:
            bestModelarity=modelarities[bestIndex]
            s[bestIndex]=-s[bestIndex]
            print(bestModelarity)
        else:
            return s

def Modelarity(s,B,m):
    eigenvalues, eigenvectors = np.linalg.eig(B)
    # Q=0
    # for i in range(len(eigenvalues)):
    #   Q=Q+np.dot(eigenvectors[i],s)**2*eigenvalues[i]/(4*m)
    Q=1/(4*m)*s.dot(B.dot(s))
    return Q

if __name__ == "__main__":

    data = []
    with open('communityExample.txt', newline='') as inputfile:
        for line in inputfile:
            if(len(line)>20):
                tmp=line[:-2].split(',')
                tmp = list(map(int, tmp))
                data.append(tmp)

    adjacency_matrix = np.array(data)
    rows, cols = np.where(adjacency_matrix == 1)
    #Måste läggas till senare, då dessa förstör resten av beräkningarna....

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
    degree_sorted = np.array([x[1] for x in degree])
    edges = gr.edges()

    color = [degree[n] for n in nodes]

    m=1/2*np.sum(degree_sorted)
    kikj=np.outer(degree_sorted,degree_sorted)
    B=adjacency_matrix-kikj/(2*m)


    eigenvalues,eigenvectors=linalg.eig(B)
    order=np.argsort(eigenvalues)[::-1]
    maxValue=np.argmax(eigenvalues)
    sortedEigenvalues=[]
    sortedEigenvectors=[]
    eigenValues = eigenvalues[maxValue]
    eigenVectors = eigenvectors[:, maxValue]
    s=np.where(eigenVectors>0,1,-1)

    s=Kernighan_Lin(s, B, m)
    group1=np.argwhere(s<0)
    group2=np.argwhere(s>0)
    numberInGroup1=len(group1)
    numberInGroup2=len(group2)

    adjacency_matrix_sub=np.zeros([numberInGroup1,numberInGroup1])

    for i in range(numberInGroup1):
        for j in range(numberInGroup1):
            adjacency_matrix_sub[i][j]=adjacency_matrix[group1[i][0]][group1[j][0]]


    degree_sub=[sum(x) for x in adjacency_matrix_sub[:]]
    B_sub=B
    dg=sum(degree_sub)
    for i in range(numberInGroup1):
        B_sub[group1[i][0]][group1[i][0]]=B_sub[group1[i][0]][group1[i][0]]-(degree_sub[i]-degree_sorted[group1[i]][0]*dg/(2*m))

    adjacency_matrix_sub2 = np.zeros([numberInGroup2, numberInGroup2])

    for i in range(numberInGroup2):
        for j in range(numberInGroup2):
            adjacency_matrix_sub2[i][j] = adjacency_matrix[group2[i][0]][group2[j][0]]

    degree_sub = [sum(x) for x in adjacency_matrix_sub2[:]]
    B_sub2 = B
    dg = sum(degree_sub)
    for i in range(numberInGroup1):
        B_sub2[group2[i][0]][group2[i][0]] = B_sub2[group2[i][0]][group2[i][0]] - (
        degree_sub[i] - degree_sorted[group2[i]][0] * dg / (2 * m))

    print(Modelarity(s,B_sub2,m))
    print(Modelarity(s,B_sub,m))
    nx.draw(gr, pos, node_size=10, node_color=s, cmap='jet', with_label=True)

    plt.show()