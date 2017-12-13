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
def Subgroup(group1,adjacency_matrix,B,degree_sorted,m):
    numberInGroup1=len(group1)

    adjacency_matrix_sub=np.zeros([numberInGroup1,numberInGroup1])
    B_sub=np.zeros([numberInGroup1,numberInGroup1])
    for i in range(numberInGroup1):
        for j in range(numberInGroup1):
            adjacency_matrix_sub[i][j] = adjacency_matrix[group1[i][0]][group1[j][0]]

    degree_sub = [sum(x) for x in adjacency_matrix_sub[:]]
    dg = sum(degree_sub)
    for i in range(numberInGroup1):
        for j in range(numberInGroup1):
            if not i==j:
                B_sub[i][j]=adjacency_matrix[group1[i][0]][group1[j][0]]-degree_sorted[group1[i][0]]*degree_sorted[group1[j][0]]/(2*m)
            else:
                B_sub[i][j] = adjacency_matrix[group1[i][0]][group1[j][0]] - degree_sorted[group1[i][0]] * degree_sorted[
                    group1[j][0]]/(2*m)-(degree_sub[i]-degree_sorted[group1[i][0]]*dg/(2*m))

    return (B_sub)

def update(B_sub,m):
    eigenvalues, eigenvectors = linalg.eig(B_sub)
    #order = np.argsort(eigenvalues)[::-1]
    maxValue = np.argmax(eigenvalues)
    maxEigenvector = eigenvectors[:, maxValue]
    s = np.where(maxEigenvector> 0, 1, -1)

    Q=Modelarity(s,B_sub,m)
    return (Q,s)

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

    m=1/2*np.sum(degree_sorted)
    kikj=np.outer(degree_sorted,degree_sorted)
    B=adjacency_matrix-kikj/(2*m)


    eigenvalues,eigenvectors=linalg.eig(B)
    order=np.argsort(eigenvalues)[::-1]
    maxValue=np.argmax(eigenvalues)

    eigenValues = eigenvalues[maxValue]
    eigenVectors = eigenvectors[:, maxValue]
    s=np.where(eigenVectors>0,1,-1)

    s=Kernighan_Lin(s, B, m)
    Q=Modelarity(s, B, m)


    group1=np.argwhere(s<0)
    group2=np.argwhere(s>0)
    B_sub=Subgroup(group1,adjacency_matrix,B,degree_sorted,m)
    B_sub2=Subgroup(group2,adjacency_matrix,B,degree_sorted,m)
    color=np.zeros(len(eigenvalues))

    (Q1,s1)=update(B_sub,m)
    (Q2,s2)=update(B_sub2,m)

    if Q1>0:
        s1 = Kernighan_Lin(s1, B_sub, m)
        Qtmp=Modelarity(s1,B_sub,m)
        for i in range(len(group1)):
            if s1[i]>0:
                color[group1[i][0]]=1
            else:
                color[group1[i][0]]=2
    print(Q1+Q)

    nx.draw(gr, pos, node_size=10, node_color=color, cmap='jet', with_label=True)
    plt.text(-0.9,1.2,'Modilarity \n %1.6f' %(Q1+Q))
    plt.show()