import numpy as np
import copy

import sklearn.cluster
from numpy.linalg import norm
from sklearn.cluster import KMeans
from scipy.constants import R
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn import *
import numpy as np

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
            rand_points = initialize_clusters(points, k_clusters)
            #print(rand_points)
            distances = compute_distances(points, rand_points, k_clusters)
        else:
            distances = compute_distances(points, centroid_points, k_clusters)
        assigned_centroids = assign_centroids(distances, k_clusters)
        calc_error = abs(objective_function_value - last_objective)
        last_objective = objective_function_value
        #print(objective_function_value)
        #print(calc_error)
        centroid_points = calculate_new_centroids(points, assigned_centroids, k_clusters)
        #print(centroid_points)
        if calc_error < error:
            print("function end with error", calc_error)
            break
        if i == n_of_iterations - 1:
            print("function end due to number of iteration with error", calc_error)

    ###################################

    return centroid_points, last_objective


def elbow_method() :
    k_all = range(2, 10)
    all_objective = []


#WRITE YOUR CODE HERE
    for k in k_all:
        fit(loaded_points, k, n_of_iterations=50, error=3)
        all_objective.append(objective_function_value)

    plt.figure()
    plt.plot(k_all, all_objective)
    plt.xlabel('K clusters')
    plt.ylabel('Sum of squared distance')
    plt.show()

#unfortunetly the elbow method is not working for this dataset,
# because it is too small difference between iterations so the elbow is not visible at the graph
#so fot this dataset is best to use either fixed number of cluster defined form graph of dataset of the silhouette score
# which implementation is byonde scope of this exercise


loaded_image = imread('fish.jpg')

plt.figure()
plt.imshow(loaded_image)
plt.show()

def compress_image(image: np.ndarray, number_of_colours: int) -> np.ndarray:
    """
    Compresses the given image by reducing the number of colours used in the image.

    This function applies k-means clustering to group the pixel colours of the image
    into 'number_of_colours' clusters. Each pixel's colour in the image is then replaced
    with the colour of the closest centroid of these clusters. This process effectively
    reduces the number of colours in the image, resulting in compression.

    Parameters:
    image (np.array): The original image is represented as a 3D numpy array
                      (height x width x color_channels).
    number_of_colours (int): The number of colours to reduce the image to.

    Returns:
    np.array: The compressed image as a numpy array in the same shape as the input.
    """
    height, width, color_channels = image.shape
    array_image = image.reshape(width*height, color_channels)

    new_colors,_ = fit(array_image, number_of_colours, n_of_iterations=100, error=0.001)

    # Create the compressed image by mapping each pixel to its centroid color
    compressed = np.zeros_like(array_image)
    centroids = new_colors.reshape(-1, number_of_colours, color_channels)
   # Reshape back to original image dimensions
    compressed_image = compressed.reshape(height, width, color_channels)

    return image




print(len(loaded_points))
plt.figure()
plt.scatter(loaded_points[:,0],loaded_points[:,1])
plt.show()

objective_function_value = 0.0
#fit(loaded_points, 5, n_of_iterations=20,error=5)
elbow_method()

img = compress_image(loaded_image, 1024)

plt.figure()
plt.imshow(img)
plt.show()