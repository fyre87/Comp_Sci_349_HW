import ID3, parse, random
import matplotlib.pyplot as plt

def testID3AndEvaluate():
    data = [dict(a=1, b=0, Class=1), dict(a=1, b=1, Class=1)]
    tree = ID3.ID3(data, 0)
    if tree != None:
        ans = ID3.evaluate(tree, dict(a=1, b=0))
        if ans != 1:
            print("ID3 test failed.")
        else:
            print("ID3 test succeeded.")
    else:
        print("ID3 test failed -- no tree returned")

def testPruning():
    # data = [dict(a=1, b=1, c=1, Class=0), dict(a=1, b=0, c=0, Class=0), dict(a=0, b=1, c=0, Class=1), dict(a=0, b=0, c=0, Class=1), dict(a=0, b=0, c=1, Class=0)]
    # validationData = [dict(a=0, b=0, c=1, Class=1)]
    data = [dict(a=0, b=1, c=1, d=0, Class=1), dict(a=0, b=0, c=1, d=0, Class=0), dict(a=0, b=1, c=0, d=0, Class=1), dict(a=1, b=0, c=1, d=0, Class=0), dict(a=1, b=1, c=0, d=0, Class=0), dict(a=1, b=1, c=0, d=1, Class=0), dict(a=1, b=1, c=1, d=0, Class=0)]
    validationData = [dict(a=0, b=0, c=1, d=0, Class=1), dict(a=1, b=1, c=1, d=1, Class = 0)]
    tree = ID3.ID3(data, 0)
    ID3.prune(tree, validationData)
    if tree != None:
        ans = ID3.evaluate(tree, dict(a=0, b=0, c=1, d=0))
        if ans != 1:
            print("pruning test failed.")
        else:
            print("pruning test succeeded.")
    else:
        print("pruning test failed -- no tree returned.")


def testID3AndTest():
    trainData = [dict(a=1, b=0, c=0, Class=1), dict(a=1, b=1, c=0, Class=1), 
    dict(a=0, b=0, c=0, Class=0), dict(a=0, b=1, c=0, Class=1)]
    testData = [dict(a=1, b=0, c=1, Class=1), dict(a=1, b=1, c=1, Class=1), 
    dict(a=0, b=0, c=1, Class=0), dict(a=0, b=1, c=1, Class=0)]
    tree = ID3.ID3(trainData, 0)
    fails = 0
    if tree != None:
        acc = ID3.test(tree, trainData)
        if acc == 1.0:
            print("testing on train data succeeded.")
        else:
            print("testing on train data failed.")
            fails = fails + 1
        acc = ID3.test(tree, testData)
        if acc == 0.75:
            print("testing on test data succeeded.")
        else:
            print("testing on test data failed.")
            fails = fails + 1
        if fails > 0:
            print("Failures: ", fails)
        else:
            print("testID3AndTest succeeded.")
    else:
        print("testID3andTest failed -- no tree returned.")	

# inFile - string location of the house data file
def testPruningOnHouseData(inFile):
    withPruning = []
    withoutPruning = []
    data = parse.parse(inFile)
    for i in range(100):
        random.shuffle(data)
        train = data[:len(data)//2]
        valid = data[len(data)//2:3*len(data)//4]
        test = data[3*len(data)//4:]
  
        tree = ID3.ID3(train, 'democrat')
        acc = ID3.test(tree, train)
        print("training accuracy: ",acc)
        acc = ID3.test(tree, valid)
        print("validation accuracy: ",acc)
        acc = ID3.test(tree, test)
        print("test accuracy: ",acc)
  
        ID3.prune(tree, valid)
        acc = ID3.test(tree, train)
        print("pruned tree train accuracy: ",acc)
        acc = ID3.test(tree, valid)
        print("pruned tree validation accuracy: ",acc)
        acc = ID3.test(tree, test)
        print("pruned tree test accuracy: ",acc)
        withPruning.append(acc)
        tree = ID3.ID3(train+valid, 'democrat')
        acc = ID3.test(tree, test)
        print("no pruning test accuracy: ",acc)
        withoutPruning.append(acc)
    print(withPruning)
    print(withoutPruning)
    print("average with pruning",sum(withPruning)/len(withPruning)," without: ",sum(withoutPruning)/len(withoutPruning))
  

def testID3_graph():
    #training_sizes
    data = parse.parse("house_votes_84.data")
    prune_acc = []
    no_prune_acc = []
    training_size_array = []
    for training_size in range(10, 300, 10):
        withPruning = []
        withoutPruning = []
        
        for i in range(100):
            random.shuffle(data)
            train = data[:int(training_size*0.6)]
            valid = data[int(training_size*0.6):training_size]
            # The traiing data will never overlap with the training data, 
            # as it is 436 observations long
            test = data[3*len(data)//4:] 
    
            tree = ID3.ID3(train, 'democrat')
            ID3.prune(tree, valid)
            acc = ID3.test(tree, test)
            withPruning.append(acc)

            tree = ID3.ID3(train+valid, 'democrat')
            acc = ID3.test(tree, test)
            withoutPruning.append(acc)
            
        training_size_array.append(training_size)
        prune_acc.append(sum(withPruning)/len(withPruning))
        no_prune_acc.append(sum(withoutPruning)/len(withoutPruning))
    plt.plot(training_size_array, no_prune_acc, label = "No Pruning Accuracy")
    plt.plot(training_size_array, prune_acc, label = "Pruning Accuracy")
    plt.legend(loc='lower center')
    plt.xlabel("Training Samples")
    plt.ylabel("Accuracy")
    plt.show()

def train_test_split(d):
    """
    in   /  d: raw .csv file OR a list of dicts
    out  /  train_d: 60% of data as list of dicts,
            test_d: 40% of data as list of dicts,
    """
    if type(d) == str:
        # If given a file path, parse it first
        d = parse.parse(d)
    train_d = d[0:int(len(d)*0.4)] 
    validation_d = d[int(len(d)*0.4):int(len(d)*0.6)] 
    test_d = d[int(len(d)*0.6):len(d)] 

    return train_d, validation_d, test_d

def test_random_forest():
    train_d, validation_d, test_d = train_test_split("candy.data")
    Tree = ID3.ID3(train_d, 0)
    Tree = ID3.prune(Tree, validation_d)
    print("Pruned Decision Tree Test Accuracy: ", ID3.test(Tree, test_d))
    Tree = ID3.ID3(train_d + validation_d, 0)
    print("Unpruned Decision Tree Test Accuracy: ", ID3.test(Tree, test_d))
    forest_acc = []
    for i in range(0, 50):
        Forest = ID3.random_forest(train_d + validation_d, 0)
        forest_acc.append(ID3.forest_test(Forest, test_d))
    
    print("Random Forest Test Accuracy: ", sum(forest_acc)/len(forest_acc))

#testID3_graph()
test_random_forest()
