import numpy as np
import copy
from numpy.linalg import norm
from sklearn.cluster import KMeans
from scipy.constants import R
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

"Load data to project"
loaded_points = np.load('k_mean_points.npy')


"define all functions"
def initialize_clusters(points: np.ndarray, k_clusters: int) -> np.ndarray:
    """
    Initializes and returns k random centroids from the given dataset.

    :param points: Array of data points.
    :type: points ndarray with shape (n, 2)

    :param k_clusters: The number of clusters to form
    :type k_clusters: int


    :return: initial_clusters
    initial_clusters: Array of initialized centroids

    :rtype:
    initial_clusters: np.array (k_clusters, 2)
    :

    """

    ###################################
    # Write your own code here #
    initial_clusters = np.array([])
    for i in range(k_clusters):
        random_index = np.random.randint(0, len(points))
        initial_clusters = np.append(initial_clusters, points[random_index])
    #print(initial_clusters)
    ###################################

    return initial_clusters


def calculate_metric(points: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """
    Calculates the distance metric between each point and a given centroid.

    Parameters:
    :param points: Array of n data points.
    :type points: ndarray with shape (n, 2)

    :param centroid: A single centroid
    :type centroid: ndarray with shape (1, 2)

    :return: distances_array
    distances_array: Array of distances from point to centroid

    :rtype:
    distances_array: ndarray with shape (n,)
    :
    """

    ###################################
    # Write your own code here #
    distances_array = np.array([])
    distances_array = norm(points - centroid, axis=1)
    #print(distances_array)

    ###################################

    return distances_array

def compute_distances(points: np.ndarray, centroids_points: np.ndarray,k_clusters: int) -> np.ndarray:
    """
    Computes and returns the distance from each point to each centroid.

    Parameters:
    :param points: Array of n data points.
    :type points: ndarray with shape (n, 2)

    :param centroids_points: A all centroid points
    :type centroids_points: ndarray with shape (k_clusters, 2)


    :return: distances_array
    distances_array: 2D array with distances of each point to each centroid.

    :rtype:
    distances_array: ndarray of shape (k_clusters, n)
    """
    ###################################
    # Write your own code here #

    distances_array = np.array([])
    for i in range(k_clusters):
        distances_array = np.append(distances_array, calculate_metric(points, centroids_points[i]))
    #print(distances_array)
    ###################################

    return distances_array

def assign_centroids(distances: np.ndarray,k_clusters: int) -> np.ndarray:
    """
    Assigns each point to the closest centroid based on the distances.

    Parameters:
    :param distances: 2D array with distances of each point to each centroid.
    :type distances: ndarray with shape (k_clusters, n)

    :return: assigned_clusters
    assigned_clusters: Array indicating the closest centroid for each data point.

    :rtype:
    assigned_centroids: ndarray with shape (1, n) and dtype = np.int32
    """

    ###################################
    # Write your own code here #
    assigned_centroids = np.array([])
    split = len(distances)//k_clusters
    split_distances = np.array_split(distances, split)
    distance_to_centroid = np.array([])
    #print(split_distances)
    for i in range(len(split_distances)):
        #print(split_distances[i])
        assigned_centroids = np.append(assigned_centroids, np.argmin(split_distances[i]))
        distance_to_centroid =np.append(distance_to_centroid,np.min(split_distances[i]))
    global objective_function_value
    objective_function_value = np.sum(distance_to_centroid)
    #print(assigned_centroids)
    #print(distance_to_centroid)


    ###################################

    return assigned_centroids

def calculate_objective(assigned_centroids: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """
    Calculates and returns the objective function value for the clustering.

    Parameters:
    :param assigned_centroids: Array indicating the cluster assignment for each point.
    :type assigned_centroids: ndarray with shape (1, n) and and dtype = np.int64

    :param distances: 2D array with distances of each point to each centroid
    :type distances: ndarray with shape (k_clusters, n) and and dtype = np.float64

    :return: onjective_function_value
    onjective_function_value: Objective function value.

    :rtype:
    onjective_function_value: float32


    This function is reduced and it's work is done in the assign_centroids function.

    """
    ###################################
    # Write your own code here #

    objective_function_value = 0.0
        #calculate the distances to the corect centroid and then I sum it op

    ###################################

    return objective_function_value


def calculate_new_centroids(points: np.ndarray, assigned_centroids: np.ndarray, k_clusters: int) -> np.ndarray:
    """
    Computes new centroids based on the current cluster assignments.

    Parameters:
    :param points: Array of n data points.
    :type points: ndarray with shape (n, 2)

    :param assigned_centroids: Array indicating the closest centroid for each data point.
    :type assigned_centroids: ndarray with shape (1, n) and dtype = np.int32


    :param k_clusters: Number of clusters.
    :type k_clusters: int


    :return: new_clusters
    new_clusters: new cluster points

    :rtype:
    new_clusters: ndarray with shape (1, n) and dtype = np.float32
    """

    ###################################
    # Write your own code here #
    new_clusters = np.array([])
    for i in range(k_clusters):
        closest_cluster =np.empty((0, 2))
        for j in range(len(assigned_centroids)):
            if assigned_centroids[j] == i:
                closest_cluster = np.vstack((closest_cluster, points[j]))
                #print(closest_cluster)
        new_clusters = np.append(new_clusters, np.average(closest_cluster, axis=0))
    #print(new_clusters)
    ###################################

    return new_clusters

def fit(points: np.ndarray, k_clusters: int, n_of_iterations: int, error: float = 0.001) -> tuple:
    """
    Fits the k-means clustering model on the dataset.

    Parameters:
    :param points : Array of data points.
    :type points: ndarray with shape (n, 2) and dtype = np.float32

    :param k_clusters:  Number of clusters
    :type k_clusters: int

    :param n_of_iterations:  Maximum number of iterations
    :type n_of_iterations: int


    :param error: Threshold for convergence.
    :type error: float

    :return: centroid_points, last_objective
    centroid_points: final centroid points
    last_objective: final objective funtion

    :rtype:
    centroid_points: ndarray with shape (k_clusters, 2) and dtype = np.float32
    last_objective: float

    """

    ###################################
    # Write your own code here #

    centroid_points = np.array([])
    last_objective = 10000.0
    calc_error = 0.001
    for i in range(n_of_iterations):
        if i == 0:
            rand_points = initialize_clusters(loaded_points, k)
            print(rand_points)
            distances = compute_distances(loaded_points, rand_points, k)
        else:
            distances = compute_distances(loaded_points, centroid_points, k)
        assigned_centroids = assign_centroids(distances, k)
        calc_error = abs(objective_function_value - last_objective)
        last_objective = objective_function_value
        print(objective_function_value)
        print(calc_error)
        centroid_points = calculate_new_centroids(loaded_points, assigned_centroids, k)
        print(centroid_points)
        if calc_error < error:
            print("function end with error", calc_error)
            break
        if i == n_of_iterations - 1:
            print("function end due to number of iteration with error", calc_error)

    ###################################

    return centroid_points, last_objective

print(len(loaded_points))
plt.figure()
plt.scatter(loaded_points[:,0],loaded_points[:,1])
plt.show()
k=3
objective_function_value = 0.0
fit(loaded_points, k, n_of_iterations=20,error=5)