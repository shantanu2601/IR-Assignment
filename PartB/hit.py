import networkx as nx
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import time
start_time = time.time()

#inverted index dictionary
dict = {}


ps = PorterStemmer()
web_graph = nx.read_gpickle("web_graph.gpickle")
num_nodes = len(web_graph.nodes)
#print(num_nodes)
#get the 50th page content
index = 0
#print(web_graph.nodes[index])

#find adjacency matrix
adj_matrix = nx.to_numpy_array(web_graph)
#find edge list
Edge_List = list(nx.edges(web_graph))
#print(adj_matrix[0,:])
stop_words = set(stopwords.words('english'))  #stop_words contains all english stop words

for node_index in range (num_nodes):
    content = web_graph.nodes[node_index]['page_content']
    word_tokens = word_tokenize(content)
    for word in word_tokens:
        word = ps.stem(word)
        if word not in stop_words:
            if word in dict:
                nodes = dict.get(word)
                if node_index not in nodes:
                    dict[word].append(node_index)
                #if dict[word][-1] cha!= node_index:
                #    dict[word].append(node_index)
            else:
                dict[word] = [node_index]    

#to make Root Set
def makeRootSet(q):
    root = []
    q = ps.stem(q)
    if q in dict:
        root = dict[q]
        return root
    print("Query word not present")
    exit()

#to make base set
def makeBaseSet(rSet):
    base_set = rSet
    for k in range((len(rSet))):
        for i in range(len(Edge_List)):
            if(Edge_List[i][0] == rSet[k]):
                if(Edge_List[i][1] not in base_set):
                    base_set.append(Edge_List[i][1])
            elif ((Edge_List[i][1] == rSet[k])):
                if(Edge_List[i][0] not in base_set):
                    base_set.append(Edge_List[i][0])
    return base_set                  


query = ""
query = input('Enter your query:\n')
rootSet = makeRootSet(query)
#print(rootSet)
baseSet = np.array(makeBaseSet(rootSet))
print("Base set:",baseSet)

def out_edges(x):
    out =np.array(list(web_graph.out_edges(x)))[:,1]
    out=np.intersect1d(out, baseSet)
    return out
#Adjacency matrix for the base set 
A=np.zeros((baseSet.shape[0],baseSet.shape[0]))
for i in range (baseSet.shape[0]):
    for j in range  (baseSet.shape[0]):
        if baseSet[j] in out_edges(baseSet[i]):
            A[i][j]=1
print("Adjacency Matrix:\n",A)               

a_old=np.ones_like(baseSet).reshape(-1,1)
h_old=np.ones_like(baseSet).reshape(-1,1)

k=10

for i in range(k):

    h_new= np.dot(A,a_old)
    h_new=h_new/np.sqrt(np.sum(h_new**2))
    a_new= np.dot(A.T,h_old)
    a_new=a_new/np.sqrt(np.sum(a_new**2))
    h_old=h_new
    a_old=a_new

print("Hub scores:",h_new.reshape(1,-1))
print("Authority scores:",a_new.reshape(1,-1))
print("Execution time : %s seconds " % (time.time() - start_time))






