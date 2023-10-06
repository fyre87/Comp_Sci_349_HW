from node import Node
import math
from parse import parse

import pprint
import pandas as pd
# import seaborn as sns
pp = pprint.PrettyPrinter(indent=4)

def getData(datasetName):
  d = parse(datasetName)
  attr_names = list(d[0].keys())
#   attr_names.remove('Class') #attr_names without target attribute Class
  return d, attr_names

def setup(examples):
  d, attr_names = getData(examples)
  
  row_ids = [row for row in range(len(d))] #list of d's row indexes
  attr_ids = [a for a in range(len(attr_names))] #list of attributes
  Y_valList = [x['Class'] for x in d]

  return d, attr_names, row_ids, attr_ids, Y_valList


def find_best_A(data, attributes):
    # Get list of As
    best_A_value = None
    best_entropy = 999
    
    for A in attributes:
        entropy = 0
        unique_A_vals = get_A_vals(data, A)
        for a in unique_A_vals:
            D_a = partition(data, a, A)
            entropy = entropy + (len(D_a)/len(data))*(get_entropy(D_a))
        
        if entropy < best_entropy:
            best_A_value = A
            best_entropy = entropy
            
    return A, attributes.index(A)

def get_A_vals(data, A):
    # Return a list of all unqiue values of A in data
    unique_vals = set()
    for i in range(0, len(data)):
        unique_vals.add(data[i][A])
    return unique_vals

def partition(data, a, A):
    # Split a dataset so that only values where A == a exist
    subset = []
    for i in range(0, len(data)):
        if data[i][A] == a:
            subset.append(data[i])
    return subset

def get_entropy(data):
    Y_valList = [x['Class'] for x in data]
    Y_valNames = list(set(Y_valList))

    # count number of instances of each label/category
    Y_valCount= {x: list(Y_valList).count(x) for x in Y_valNames} #counts in each category

    #calculate frequencies aka probabilities
    total = len(data) #of total examples
    freqs = [Y_valCount[x]/total for x in Y_valNames]
    # print(freqs)

   # calculate the entropy for each category + adds
    entropy = sum([-freq * math.log(freq, 2) if freq else 0 for freq in freqs])
    return entropy


def xyz():
    d, attr_names, row_ids, attr_ids, Y_valList = setup('candy.data')
    
    t1= get_entropy(d)

    # pp.pprint(y)
    pp.pprint(Y_valList)
    pp.pprint(len(d))
    
# xyz()

def ID3(examples): #first pass -- review 
    d, attr_names, row_ids, attr_ids, Y_valList = setup(examples)
    attr_names.remove('Class') #remove class from attribute name list
    # node = ID3(d, attr_names, Y_valList)
    
    node = Node(d)
    node.label = max(set(Y_valList), key=Y_valList.count)  #setting node label to most common label

    # if pure node (all the data in node have the same class)
    if node.is_pure() == True:
        node.label = node.find_most_common() #node.label or node.value?
        return node
    
    #if no more features to compute, attribute list empty
    if len(attr_names) == 0:
        #set to most common class in the entire dataset
        node.label = max(set(Y_valList), key=Y_valList.count)  #node.label or node.value?
        return node
    
    bestA, bestA_index = find_best_A(d, attr_names)
    node.value = bestA
    bestA_vals = get_A_vals(d, bestA)
    print(bestA_vals)
    for v in bestA_vals:
        child = Node(v)
        node.children.append(child)  # append new child node to current node
        D_v = partition(d, v, bestA)
        Y_v = [x['Class'] for x in D_v]
        if len(D_v) == 0:
            max(set(Y_valList), key=Y_valList.count)  #set to most common class in the entire dataset
        else:
            attr_names.pop(bestA_index)
            ID3(D_v, attr_names, Y_v)

ID3('candy.data')
