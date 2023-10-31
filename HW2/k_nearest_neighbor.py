import numpy as np 
import math
# from starter import euclidean, cosim
# from distance import euclidean_distances, manhattan_distances

# returns Euclidean distance between vectors a dn b
def euclidean(a,b):
    # da = [int(i) for i in a]
    # db = [int(i) for i in b]

    #using math
    # dist1 = math.dist(a, b)

    # #using numpy
    d1, d2 = np.array(a), np.array(b)
    dist2 = np.sqrt(np.sum((d1 - d2)**2))

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
    #dist1 = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    # print(dist,dist1)
    return(dist)

class KNearestNeighbor():    
    def __init__(self, n_neighbors, distance_measure='euclidean', aggregator='mode'):
        """
        K-Nearest Neighbor is a straightforward algorithm that can be highly
        effective. Training time is...well...is there any training? At test time, labels for
        new points are predicted by comparing them to the nearest neighbors in the
        training data.

        ```distance_measure``` lets you switch between which distance measure you will
        use to compare data points. The behavior is as follows:

        If 'euclidean', use euclidean_distances, if 'manhattan', use manhattan_distances.

        ```aggregator``` lets you alter how a label is predicted for a data point based 
        on its neighbors. If it's set to `mean`, it is the mean of the labels of the
        neighbors. If it's set to `mode`, it is the mode of the labels of the neighbors.
        If it is set to median, it is the median of the labels of the neighbors. If the
        number of dimensions returned in the label is more than 1, the aggregator is
        applied to each dimension independently. For example, if the labels of 3 
        closest neighbors are:
            [
                [1, 2, 3], 
                [2, 3, 4], 
                [3, 4, 5]
            ] 
        And the aggregator is 'mean', applied along each dimension, this will return for 
        that point:
            [
                [2, 3, 4]
            ]

        Arguments:
            n_neighbors {int} -- Number of neighbors to use for prediction.
            distance_measure {str} -- Which distance measure to use. Can be one of
                'euclidean' or 'manhattan'. This is the distance measure
                that will be used to compare features to produce labels. 
            aggregator {str} -- How to aggregate a label across the `n_neighbors` nearest
                neighbors. Can be one of 'mode', 'mean', or 'median'.
        """
        self.n_neighbors = n_neighbors
        self.distance_measure = distance_measure
        self.aggregator = aggregator
        self.features = None #initialize with fit()
        self.labels = None # intialize with fit()

    def fit(self, features, targets):
        """Fit features, a numpy array of size (n_samples, n_features). For a KNN, this
        function should store the features and corresponding targets in class 
        variables that can be accessed in the `predict` function. Note that targets can
        be multidimensional! 
        
        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            targets {[type]} -- Target labels for each data point, shape of (n_samples, 
                n_dimensions).
        """
        self.features = np.where(features <=128, 0, 1)
        self.labels = targets        

    def predict_help(self, example_features):
        """uses knn to classify ONE example
        returns mode label 
        """
        distList = []
        
        for i in range(len(self.features)):
            if self.distance_measure == 'euclidean':
                distList.append((self.labels[i], euclidean(self.features[i], example_features)))
                #sorting by lowest distance x[1] for euclidean
                distList.sort(key = lambda x: x[1])
            elif self.distance_measure == 'cosim':
                distList.append((self.labels[i], cosim(self.features[i], example_features)))
                #sorting by highest distance x[1] for cosim
                distList.sort(key = lambda x: x[1], reverse = True)
            
            #getting just the labels i[0], distances are i[1]
            k_nns = [i[0] for i in distList[:self.n_neighbors]]
        
        # if self.aggregator == 'mode':
        #     probLabel = max(set(k_nns), key = k_nns.count) # 3
        # elif self.aggregator == 'median':
        #     probLabel = np.median(k_nns, axis = 0)
        # elif self.aggregator == 'mean':
        #     probLabel = np.mean(k_nns, axis = 1)
        probLabel = max(set(k_nns), key = k_nns.count)
        return probLabel
    
    def predict(self, query_features, ignore_first=False):
        """Predict from features, a numpy array of size (n_samples, n_features) Use the
        training data to predict labels on the test features. For each testing sample, compare it
        to the training samples. Look at the self.n_neighbors closest samples to the 
        test sample by comparing their feature vectors. The label for the test sample
        is the determined by aggregating the K nearest neighbors in the training data.

        Note that when using KNN for imputation, the predicted labels are the imputed testing data
        and the shape is (n_samples, n_features).

        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            ignore_first {bool} -- If this is True, then we ignore the closest point
                when doing the aggregation. This is used for collaborative
                filtering, where the closest point is itself and thus is not a neighbor. 
                In this case, we would use 1:(n_neighbors + 1).

        Returns:
            labels {np.ndarray} -- Labels for each data point, of shape (n_samples,
                n_dimensions). This n_dimensions should be the same as n_dimensions of targets in fit function.
        """
        query_features = np.where(query_features <=128, 0, 1)
        labels = [self.predict_help(i) for i in query_features]
        return labels


