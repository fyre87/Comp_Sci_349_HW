from node import Node
import math

import math

def find_best_A(data):
    # Get list of A's
    attributes = []
    for i in data[0].items():
        if i[0] != 'Class':
            attributes.append(i[0])
            
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
            
    return A

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
    #Dictionary to store the instances of each class in a dataset
    ratios = {}
    for i in range(0, len(data)):
        if data[i]['Class'] not in ratios:
            # If not yet in dictionary, add it!
            ratios[data[i]['Class']] = 1
        else:
            # If in dictionary, add 1!
            ratios[data[i]['Class']] = ratios[data[i]['Class']] + 1
            
    items = ratios.items()

    entropy_sum = 0
    # Now loop through classes
    for item in items:
        #entropy_sum = entropy_sum + (num of class i / whole data) * (whole_entropy)
        entropy_sum = entropy_sum + (entropy_helper(item[1]/len(data)))
    return entropy_sum
        
        
def entropy_helper(p):
    return -1*p*math.log2(p)
            







def ID3(examples, default):
  '''
  Takes in an array of examples, and returns a tree (an instance of Node) 
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
  '''
  print(find_best_A(examples))

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

