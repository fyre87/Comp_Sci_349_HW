import math
import numpy as np
import csv 
from random import sample

# returns Euclidean distance between vectors a dn b
def euclidean(a,b):
    # da = [int(i) for i in a]
    # db = [int(i) for i in b]

    #using math
    # dist1 = math.dist(a, b)

    # #using numpy
    # d1, d2 = np.array(a), np.array(b)
    # dist2 = np.sqrt(np.sum((d1 - d2)**2))

    #naive
    dist = math.sqrt(sum([(a_x - b_x)**2 for a_x, b_x in zip(a,b)]))
    # print(dist, type(dist))
    # print(dist1, type(dist1))
    # print(dist2, type(dist2))

    return(dist)
        
# returns Cosine Similarity between vectors a dn b

def dot(x,y):
    return sum(x1*x2 for x1,x2 in zip(x,y))
def cosim(a,b):
    #naive
    dist = dot(a,b) / (math.sqrt(dot(a,a)) * math.sqrt(dot(b,b)))

    #check vs numpy
    dist1 = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    # print(dist,dist1)
    return(dist)


# returns a list of labels for the query dataset based upon labeled observations in the train dataset.
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.

#helper function, given 1 example (query_i), what are the k nearest neighbors' LABELS?
def getKnns(trainSet, query_i, k, metric):
    distList = [] #append tuples of label, distance computed
    for label, val in trainSet:
        if metric == 'euclidean':
            distList.append((label, euclidean(val, query_i[1])))
        elif metric == 'cosim':
            distList.append((label, cosim(val, query_i[1])))
    #sorting by lowest distance x[1]
    distList.sort(key = lambda x: x[1])
    #getting just the labels i[0], distances are i[1]
    k_nns = [i[0] for i in distList[:k]]
    return k_nns

#helper: given k nearest neighbors, what is the probability of the query_i's class
def mostprobLabel(k_nns):
    probLabel = max(set(k_nns), key = k_nns.count) # 3
    return probLabel

def knn(train,query,metric):
    #hyper-parameters
    k = 3
    num_obs = 300
    actual_labels = []
    labels = []
    for q in query:
        nns = getKnns(train, q, k, metric)
        q_label = mostprobLabel(nns)
        actual_labels.append(q[0])
        labels.append(q_label)
    print(actual_labels, labels)

# returns a list of labels for the query dataset based upon observations in the train dataset. 
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def kmData(data_set):
    km_D = [item[1] for item in data_set]
    return km_D

def kmeans(train,query,metric):
    k = 3
    max_iters = 3

    #initiating centroids randomly
    centroids = sample(train, k)

    for i in range(max_iters):
        groups = {}
        distList = []
        for i in range(k):
            groups[i] = []
        for t in train:
            distList = [euclidean(t, centroids[c]) for c in centroids]
            t_Group = distList.index(min(distList))
            groups[t_Group].append(t)

    return(labels)

def read_data(file_name):
    data_set = []
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label,attribs])
    return(data_set)
        
def read_data_COMP(file_name):
    data_set = []
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(',')
            label = int(tokens[0])
            attribs = []
            for i in range(784):
                #TODO: these 3 lines try to fix train.csv error
                if tokens[i+1] == '':
                    attribs.append(0)
                else: 
                    attribs.append(int(tokens[i+1]))
            data_set.append([label,attribs])
    return(data_set)

def show(file_name,mode):
    data_set = read_data(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ',end='')
                else:
                    print('*',end='')
            else:
                print('%4s ' % data_set[obs][1][idx],end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0],end='')
        print(' ')
            
def main():
    d = read_data_COMP('test.csv')
    query = d[1:5]
    testset = d[6:100]
    res = knn(testset, query, 'euclidean')
    print(res)


    # print(d)
    # tlist = []
    # for label, vals in d[:5]:
    #     print("--label is ", label)
    #     print("--data is ,", vals)
    #processing data
    d_0 = d[0][1]
    d_1 = d[1][1]
    # print(d_0)

    # --- converting the num from strings to ints
    # x = [int(i) for i in d_0]
    # y = [int(i) for i in d_1]

    # print(x)
    # euc = euclidean(d_0,d_1)
    # csim = cosim(d_0,d_1)
    # print(euc, csim)

    # show('valid.csv','pixels')
    
if __name__ == "__main__":
    main()
    