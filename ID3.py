from node import Node
import math
import random
from parse import parse

def getData(d):
    """
    in   /  d: raw .csv file OR a list of dicts
    out  /  training_d: 80% of data as list of dicts,
            validation_d: 20% of data as list of dicts,
            attr_names: list of attribute names (including target attr)
            most_common: calculates most common 'Class' (aka target attr) of entire dataset
    """
    if type(d) == str:
        # If given a file path, parse it first
        d = parse(d)
    training_d = d[0:int(len(d)*0.8)] #Take 80% for training
    validation_d = d[int(len(d)*0.8):len(d)] #Take 20% for validation
    #assert len(training_d) + len(validation_d) == len(d)
    attr_names = list(d[0].keys())
    Y_valList = [x['Class'] for x in d]
    most_common = max(set(Y_valList), key=Y_valList.count) 
    #   attr_names.remove('Class') #attr_names without target attribute Class
    return training_d, validation_d, attr_names, most_common

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

    return best_A_value, attributes.index(best_A_value)

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

   # calculate the entropy for each category + adds
    entropy = sum([-freq * math.log(freq, 2) if freq else 0 for freq in freqs])
    return entropy

#------------------------
def most_common_class(d):
    Y_valList = [x['Class'] for x in d]
    most_common = max(set(Y_valList), key=Y_valList.count) 
    return most_common

def ID3_helper(data, attr_names, targ):
    # d, attr_names = getData()
    if 'Class' in attr_names:
        attr_names.remove('Class') #remove class from attribute name list

    node = Node(data)
    node.label = most_common_class(data) #setting node label to most common label

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
    
    for v in bestA_vals:
        # child = Node(v)
        # node.children[v] = child  # append new child node to current node
        
        D_v = partition(data, v, bestA)

        if len(D_v) == 0:
            child = Node([])
            child.label = targ #set to most common class in the entire dataset
            child.is_leaf = True
            node.children[v] = child
        else:
            new_attr_names = attr_names[0:bestA_index] + attr_names[bestA_index + 1:]
            node.children[v] = ID3_helper(D_v, new_attr_names, targ)

    return node

def prune(val_data, DT):
    for d in val_data:
        pruning_inf(d, DT)
    prune_help(DT)
    return DT

def pruning_inf(d, DT):
    """
    Adds to the "count_correct" and "count_incorrect" for each node d is on
    """
    if DT.label == d['Class']:
        DT.count_correct += 1
    else:
        DT.count_incorrect += 1

    if DT.is_leaf == True:
        # At a leaf node. Just add to the count and stop
        return
    
    A = DT.split_attribute

    if d[A] == '?':
        # If the value is missing, can't possibly go forward. So just stop
        return
    
    # Follow the child
    # Make sure the child is actually in the dictionary. 
    # If there is a non seen value in the validation set, can't continue
    if d[A] in DT.children:
        pruning_inf(d, DT.children[d[A]])
    else:
        return

def prune_help(DT):
    if DT.is_leaf == True:
        return
    
    if DT.count_correct + DT.count_incorrect == 0:
        # Validation set didn't get here. Probably not reliable, prune it
        DT.children = {}
        DT.is_leaf = True
        return
    
    parent_acc = (DT.count_correct)/(DT.count_correct + DT.count_incorrect)
    child_acc = 0

    # Note: The total count in the children may not be the same
    # as the count in the parent due to missing data
    # Thus we must gather the total in the children first!
    children_observation_count = 0
    
    for k in DT.children.keys():
        y = DT.children[k].count_correct + DT.children[k].count_incorrect
        children_observation_count += y

    # Could have the child observation count be 0 while parent observation count is nonzero
    # due to a very rare edge case where the validation data has a split value
    # that the training data doesn't have. Thus only continue if child_observation_count > 0
    if children_observation_count != 0:
        for k in DT.children.keys():
            x = DT.children[k].count_correct
            y = DT.children[k].count_correct + DT.children[k].count_incorrect
            if y != 0:
                child_acc += (x/y)*(y/children_observation_count)
    

    if child_acc < parent_acc:
        #Prune
        DT.children = {}
        DT.is_leaf = True
        return
    else:
        for k in DT.children.keys():
            prune_help(DT.children[k])

    
    

def ID3(examples, default):
    '''
    Takes in an array of examples, and returns a tree (an instance of Node) 
    trained on the examples.  Each example is a dictionary of attribute:value pairs,
    and the target class variable is a special attribute with the name "Class".
    Any missing attributes are denoted with a value of "?"
    '''

    training_d, validation_d, attr_names, most_common = getData(examples) #initializing variables

    Tree = ID3_helper(training_d, attr_names, most_common)
    Tree = prune(validation_d, Tree)
    return Tree


def test(DT, test_data):
    '''
    Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
    of examples the tree classifies correctly).
    '''
    correct = 0
    incorrect = 0
    for i in range(0, len(test_data)):
        if evaluate(DT, test_data[i]) == test_data[i]['Class']:
            correct += 1
        else:
            incorrect += 1
    if correct + incorrect == 0:
        return 0
    else:
        return (correct / (correct + incorrect))

def evaluate(DT, d):
    '''
    Takes in a tree and one example.  Returns the Class value that the tree
    assigns to the example.
    '''
    if DT.is_leaf == True:
        return DT.label
    
    A = DT.split_attribute

    if d[A] == '?':
        # Missing data, just return majority class at current node
        return DT.label
    
    if d[A] not in DT.children:
        # More missing data
        return DT.label
    
    return evaluate(DT.children[d[A]], d)

###############################
### Add Random Forest Below ###
###############################


def train_test_split(d):
    """
    in   /  d: raw .csv file OR a list of dicts
    out  /  train_d: 70% of data as list of dicts,
            test_d: 30% of data as list of dicts,
    """
    if type(d) == str:
        # If given a file path, parse it first
        d = parse(d)
    train_d = d[0:int(len(d)*0.7)] #Take 80% for training
    test_d = d[int(len(d)*0.7):len(d)] #Take 20% for validation

    return train_d, test_d


def random_forest_sample(data):
    """
    Create a sample with replacement of 50% of the dataset
    and square root of features number of features. 
    """
    attribute_names = list(data[0].keys())
    attribute_names.remove('Class')

    sample = []
    attribute_list = random.choices(attribute_names, k = int(len(attribute_names)**0.5))
    attribute_list.append('Class')
    smaller_data = keep_certain_attributes(data, attribute_list)
    for j in range(0, int(len(data)*0.5)):
        # Select the attributes you want
        sample.append(smaller_data[random.randint(0, len(smaller_data)-1)])
    return sample



def keep_certain_attributes(data, attribute_list):
    """
    Takes a dataset and attributes and removes all attributes not in the list. 
    """
    for i in range(0, len(data)):
        data[i] = {key: data[i][key] for key in attribute_list}
    return data

def random_forest(data, default):
    forest = []
    for i in range(0, 100):
        #Append this 100 times
        sample = random_forest_sample(data)
        Tree = ID3(sample, 0)
        forest.append(Tree)
    return forest

def random_forest_evaluate(forest, d):
    predictions = []
    for i in range(0, len(forest)):
        # Evaluate with each forest
        predictions.append(evaluate(forest[i], d))
    return max(set(predictions), key = predictions.count)

def forest_test(forest, test_data):
    correct = 0
    incorrect = 0
    for i in range(0, len(test_data)):
        if random_forest_evaluate(forest, test_data[i]) == test_data[i]['Class']:
            correct += 1
        else:
            incorrect += 1
    if correct + incorrect == 0:
        return 0
    else:
        return (correct / (correct + incorrect))


train_d, test_d= train_test_split("candy.data")
print(test(ID3(train_d, 0), test_d))
print(forest_test(random_forest(train_d, 0), test_d))


