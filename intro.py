import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial.distance import cdist
import matplotlib.cm as cm
from pyclustering.cluster.bsas import bsas_visualizer, bsas


#######################################################################################################################
# Function plot_random(n_samples,centers):
# Function generate and plot the random data
#
# Input arguments:
#   - int n_samples: no. of samples
#   - list centers: list of center points  hint: centers=[[-2, -1],[4,4], [1, 1]] 
# Output argument:
#   - X: Array of shape [n_samples, n_features]. (Feature Matrix)
#######################################################################################################################
def random_data(n_samples=2000,centers=[[-2, -1],[4,4], [1, 1]]):
    
    #set up a random seed to zero
    np.random.seed(0)

    X = make_blobs(n_samples=n_samples, centers=centers, n_features=2)
    plt.plot(X[0], X[1])
    plt.show()

    #TODO:
    # -generate the random data using n_samples and center parameter values (hint: use make_blobs())
    # plot the ras data
    # return X: Array of shape [n_samples, n_features]. (Feature Matrix)

    return X
 
  #######################################################################################################################
# Function K_mean(clusters,X):
# Function apply the k-mean on the data and generate the plot
#
# Input arguments:
#   - int clusters: no. of clusters
#   - X: Array of shape [n_samples, n_features]. (Feature Matrix)
# Output argument:
#   - k_means_cluster_centers : 
#   - k_means_labels:

#######################################################################################################################

def k_mean(nb_clusters,X):
    
    #TODO:

     # Implement the K-Mean clustering algoritm
     # Fit the KMean algorithm
     # get the labels and cluster centers

    kmeans = KMeans(n_clusters=nb_clusters).fit_predict(X)
    k_means_labels = kmeans.labels
    k_means_cluster_centers = kmeans.cluster_centers_
    
    # Specify the plot with dimensions.
    fig = plt.figure(figsize=(6, 4))

    # Colors uses a color map, which will produce an array of colors based on
    # the number of labels there are. We use set(k_means_labels) to get the
    # unique labels.
    colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

    # Create a plot
    ax = fig.add_subplot(1, 1, 1)

    # For loop that plots the data points and centroids.
    # k will range from 0-3, which will match the possible clusters that each
    # data point is in.
    for k, col in zip(range(len(k_means_cluster_centers)), colors):

        # Create a list of all data points, where the data poitns that are 
        # in the cluster (ex. cluster 0) are labeled as true, else they are
        # labeled as false.
        my_members = (k_means_labels == k)
    
        # Define the centroid, or cluster center.
        cluster_center = k_means_cluster_centers[k]
    
        # Plots the datapoints with color col.
        ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    
        # Plots the centroids with specified color, but with a darker outline
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

    # Set title of the plot
    ax.set_title('K-Means')
    
    #Set the x-axis label
    ax.set_xlabel('x-axis')

    #Set the x-axis label
    ax.set_ylabel('y-axis')

    # Show the plot
    plt.show()
    return k_means_cluster_centers, k_means_labels

#######################################################################################################################
# Function Plot_elbow(clusters,X):
# Function apply the k-mean on the data and generate the plot
#
# Input arguments:
#   - int clusters: no. of clusters
#   - X: Array of shape [n_samples, n_features]. (Feature Matrix)

#######################################################################################################################

def plot_elbow(q,X):
    # to determine the optimal k
    distortions = [] #stores the average of the squared distances from the cluster centers of the respective clusters
    K = range(1,q)
    for k in K:
        kmean_model = KMeans(n_clusters=k).fit(X)
        kmean_model.fit(X)
        k_means_cluster_centers = kmean_model.cluster_centers_
        #TODO  
        #Compute the average of the squared distances from the cluster centers of the respective clusters using the Euclidean distance metric
        distortions[k] = cdist(k_means_cluster_centers[k], q, 'euclidean')

    pass

    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

#######################################################################################################################
# Function BSAS(X,max_clusters,threshold):
# Function Implement Basic Sequential Algorithmic Scheme (as in lecture slides 220111) and plot the data
# hint:  install the pyclustering library (i.e pip install pyclustering)
# Input arguments:
#   - array of data hint. random generated samples
#   - float threshold: default value 1.0
#   - int max_clusters: maximum clusters
#######################################################################################################################

def BSAS(X, max_clusters = 3, threshold = 1.0):

    #TODO
    # Create instance of BSAS algorithm.
    # initiate the bsas instance process
    
    # Get the clustering results of BSAS and assign it to a valriable clusters.
    # Get the representatives  of bsas instance and assign it to a variavle representatives
    bsas_instance = bsas(X, max_clusters, threshold)
    bsas_instance.process()
    clusters = bsas_instance.get_clusters()
    representatives = bsas_instance.get_representatives()
    # Display the results of BSAS.
    bsas_visualizer.show_clusters(X, clusters, representatives)

#######################################################################################################################
# Function silhoute_analysis(X, cluster_centers labels,n_clusters):
# Function perform the silhoute analysis using the data and generate silhoute plot
#
# Input arguments:
#      - X: Array of shape [n_samples, n_features]. (Feature Matrix)
#      - cluster_centers: k_mean cluster center
#      - labels : k-mean labels
#      - nb_cluster: no. of clusters

#######################################################################################################################
def silhoute_analysis(X,cluster_centers,labels,n_clusters=3):
    
    
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    # TODO
    # silhouette_avg for all the samples (hint: use silhouette_score function).
    # Print the silhouette_avg for nb_clusters
    # Compute the silhouette scores for each sample (hint: use silhouette_samples function )
    pass
    
    y_lower = 10
     # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    for i in range(n_clusters):
            
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
        sample_silhouette_values[labels == i]
    
        ith_cluster_silhouette_values.sort()
    
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
    
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
    
        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("silhouette plot.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='r')

    # Labeling the clusters
    centers = cluster_centers
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='r')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("cluster visualization.")
    plt.suptitle(("Silhouette analysis"
                  "with  n_clusters  = %d" % n_clusters))

    plt.show()



if __name__ == '__main__':
    
    
    n_samples=2000
    X=random_data(n_samples) #random data with 2000 samples

    nb_clusters=3
    cluster_centers,labels=k_mean(nb_clusters,X)  # shows the results of k-mean ckustering

    plot_elbow(10,X)  #Plot the elbow to determine the no. of clusters.

    #BSAS(X) #  Analysis of random data using Basic Sequential Algorithmic Scheme(BSAS).

    #silhoute_analysis(X,cluster_centers,labels) # Analysis of random data  using silhoute score.




    
    
    
    
    
    
   


   