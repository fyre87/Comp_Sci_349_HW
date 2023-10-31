import numpy as np
import math
import random

#Euclidean
def euclidean(a,b):
    #naive
    dist = math.sqrt(sum([(a_x - b_x)**2 for a_x, b_x in zip(a,b)]))
    return(dist)

def dot(x,y):
    return sum(x1*x2 for x1,x2 in zip(x,y))
def cosim(a,b):
    #naive
    dist = dot(a,b) / (math.sqrt(dot(a,a)) * math.sqrt(dot(b,b)))

    #check vs numpy
    #dist1 = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return(dist)

class KMeans():
    def __init__(self, n_clusters, distance_measure = 'euclidean'):
        """
        This class implements the traditional KMeans algorithm with hard assignments:

        https://en.wikipedia.org/wiki/K-means_clustering

        The KMeans algorithm has two steps:

        1. Update assignments
        2. Update the means

        While you only have to implement the fit and predict functions to pass the
        test cases, we recommend that you use an update_assignments function and an
        update_means function internally for the class.

        Use only numpy to implement this algorithm.

        Args:
            n_clusters (int): Number of clusters to cluster the given data into.

        """
        self.distance_measure = distance_measure
        self.n_clusters = n_clusters
        self.means = None

    def fit(self, features):
        """
        Fit KMeans to the given data using `self.n_clusters` number of clusters.
        Features can have greater than 2 dimensions.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            None (saves model - means - internally)
        """
        features = np.where(features <=128, 0, 1)
        self.means = self.get_random_means(features)
        while True:
            labels = self.calculate_distances(self.means, features)
            new_means = self.recalculate_means(labels, features)
            if np.array_equal(new_means, self.means):
                break
            self.means = new_means



    def get_random_means(self, features):
        if self.n_clusters > len(features):
            raise ValueError("n_clusters is larger than the number of rows!")
        return features[random.sample(range(len(features)), self.n_clusters)]
        #return np.random.rand(len(features), self.n_clusters)

    def calculate_distances(self, means, features):
        # Create distances matrix which we will fill
        # Rows are observations, 
        # each column is the distance between that cluster and the mean
        distances = np.zeros((len(features), self.n_clusters))
        for i in range(0, len(features)):
            for k in range(0, len(means)):
                if self.distance_measure == "euclidean":
                    distances[i][k] = euclidean(features[i], means[k])
                elif self.distance_measure == "cosim":
                    distances[i][k] = cosim(features[i], means[k])

        # Now find the closest point for each
        labels = np.zeros((len(features)))
        for i in range(0, len(distances)):
            if self.distance_measure == "euclidean":
                labels[i] = distances[i].argmin()
            elif self.distance_measure == "cosim":
                labels[i] = distances[i].argmax()
        return labels

    def recalculate_means(self, labels, features):
        new_means = np.zeros((self.n_clusters, len(features[0])))
        for k in range(0, self.n_clusters):
            update = features[labels == k].mean(axis = 0)
            
            if update.shape[0] == 0:
                # If had no labels, just 
                new_means[k] = self.means[k]
            else:
                new_means[k] = update
        return new_means
        
    def predict_help(self, example_features):
        """uses knn to classify ONE example
        returns closest mean label
        """
        distList = []
        
        for i in range(len(self.means)):
            if self.distance_measure == 'euclidean':
                distList.append((i, euclidean(self.means[i], example_features)))
                #sorting by lowest distance x[1] for euclidean
                distList.sort(key = lambda x: x[1])
            elif self.distance_measure == 'cosim':
                distList.append((i, cosim(self.means[i], example_features)))
                #sorting by highest distance x[1] for cosim
                distList.sort(key = lambda x: x[1], reverse = True)
        return distList[0][0]


    def predict(self, features):
        """
        Given features, an np.ndarray of size (n_samples, n_features), predict cluster
        membership labels.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            predictions (np.ndarray): predicted cluster membership for each features,
                of size (n_samples,). Each element of the array is the index of the
                cluster the sample belongs to.
        """
        features = np.where(features <=128, 0, 1)
        labels = [self.predict_help(i) for i in features]
        return labels