from node import Node
import math
from parse import parse

def getData(datasetName):
    """
    in   /  datasetName: raw .csv file
    out  /  d: data as list of dicts, 
            attr_names: list of attribute names (including target attr)
            most_common: calculates most common 'Class' (aka target attr) of entire dataset
    """
    d = parse(datasetName)
    attr_names = list(d[0].keys())
    Y_valList = [x['Class'] for x in d]
    most_common = max(set(Y_valList), key=Y_valList.count) 
#   attr_names.remove('Class') #attr_names without target attribute Class
    return d, attr_names, most_common

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

#------------------------
def main(rawdata): #TODO: remake to be ID3 function?
    d, attr_names, most_common = getData(rawdata) #initializing variables
    ID3_helper(d, attr_names, most_common)
    return

def ID3_helper(data, attr_names, targ):
    # d, attr_names = getData()
    attr_names.remove('Class') #remove class from attribute name list
    
    node = Node(data)
    majority_label = targ
    node.label = majority_label #setting node label to most common label

    # if pure node (all the data in node have the same class)
    if node.is_pure() == True:
        node.is_leaf = True
        return node
    
    #if no more features to compute, attribute list empty
    if len(attr_names) == 0:
        node.is_leaf = True
        return node
    
    bestA, bestA_index = find_best_A(data, attr_names)
    node.split_attribute = bestA
    bestA_vals = get_A_vals(data, bestA)
    # print(bestA_vals)
    
    for v in bestA_vals:
        # child = Node(v)
        # node.children[v] = child  # append new child node to current node
        
        D_v = partition(data, v, bestA)

        if len(D_v) == 0:
            child = Node([])
            child.label = majority_label #set to most common class in the entire dataset
            child.is_leaf = True
            node.children[v] = child
        else:
            attr_names.pop(bestA_index)
            node.children[v] = ID3_helper(D_v, attr_names, majority_label)

    return node

def ID3(examples, default):
  '''
  Takes in an array of examples, and returns a tree (an instance of Node) 
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
  '''

def prune(node, examples):
  '''
  Takes in a trained tree and a validation set of examples.  Prunes nodes in order
  to improve accuracy on the validation data; the precise pruning strategy is up to you.
  '''

def test(node, examples):
  '''
  Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).
  '''



def evaluate(node, example):
  '''
  Takes in a tree and one example.  Returns the Class value that the tree
  assigns to the example.
  '''


main('candy.data')
