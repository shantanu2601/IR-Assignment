import numpy as np
import time



n = int(input("Number of nodes: "))
e = int(input("Number of edges: "))

edges=[]
edge_from_node=[0]*n
L = [ [ 0 for i in range(n) ] for j in range(n) ]

rank = [1/n]*n
rank = np.array(rank)

''' making the adjecency matrix'''
for i in range(e):
    a = input()
    edges.append(tuple(int(x) for x in a.split(",")))
    edge_from_node[edges[i][0]-1] += 1
    L[edges[i][0]-1][edges[i][1]-1] += 1

start_time = time.time()

for i in range(n):
    if edge_from_node[i] !=0:
        L[i][:] = [x/edge_from_node[i] for x in L[i]] 

L = np.array(L)

def randTele(L):
    '''
    Implements the Random teleportation with a probablity 0.1\n
    Takes adjacency matrix (list[list[]]) as parameter\n
    Return the adjacency Matrix with random teleportaiton
    '''
    rand_tele = [ [ 1/n for i in range(n) ] for j in range(n) ]
    rand_tele = np.array(rand_tele)
    L = L*0.9 + rand_tele*0.1
    return L


def pack(T):
    '''
    pack method claculates the left eigenvector using np.linalg.eig()\n
    Takes adjacency matrix (list[list[]]) as the parameter\n
    Returns Rank vector
    '''
    probMat = randTele(T)
    eigvalues, eigvectors = np.linalg.eig(probMat.T)
    left_vec = eigvectors[:, np.argmax(eigvalues)].T
 
    print("Rank vector using pakacges: ",left_vec/sum(left_vec))

def powerItr(L,rank):
    '''
    powerItr method calculates left eigen vector using power iteration method\n
    Takes adjacency matrix (list[list[]]) and rank vector (list[]) as parameters\n
    Loop runs till the difference betweeen the norm of two vectors < 10e-9\n
    Returns rank vector
    '''
    count = 0
    L = randTele(L)
    while(1):
        if (np.linalg.norm(rank-np.dot(rank,L)))<0.000000001:
            break;    
        rank = np.dot(rank,L)
        count +=1
    print("Rank vector using power iteration: ",rank)
    #print(count)


powerItr(L,rank)
pack(L)

print("Runing time: ",time.time()-start_time)
