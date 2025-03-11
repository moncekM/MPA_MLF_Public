"Inport of libraries"
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
    #print(split_distances)
    for i in range(len(split_distances)):
        assigned_centroids = np.append(assigned_centroids, np.argmin(split_distances[i]))
    #print(assigned_centroids)
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


    """
    ###################################
    # Write your own code here #

    objective_function_value = 0.0
    for i in range(len(assigned_centroids)):
        objective_function_value += (np.sum(distances[i] - assigned_centroids) ** 2)

    ###################################

    return objective_function_value

print(len(loaded_points))
plt.figure()
plt.scatter(loaded_points[:,0],loaded_points[:,1])
plt.show()
k=3
rand_points=initialize_clusters(loaded_points,k)
print(rand_points)
cluster_points = np.array([])
#cluster_poitrn = calculate_metric(loaded_points, rand_points[1])
distances = compute_distances(loaded_points, rand_points,k)
#print(distances)
assigned_centroids = assign_centroids(distances,k)
print(assigned_centroids)
print(len(assigned_centroids))
objective_function_value = calculate_objective(assigned_centroids, distances)
print(objective_function_value)
