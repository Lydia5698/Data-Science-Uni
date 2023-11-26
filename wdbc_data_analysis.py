import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from numpy import array
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# For reproducibility
np.random.seed(1000)

bc_data_path = 'wdbc.data'  # path to the input data file.

bc_data_columns = ['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
                   'compactness_mean', 'concavity_mean',
                   'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se',
                   'perimeter_se', 'area_se', 'smoothness_se',
                   'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
                   'radius_worst', 'texture_worst', 'perimeter_worst',
                   'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst',
                   'symmetry_worst', 'fractal_dimension_worst']


#######################################################################################################################
# Function correlation_map(dataExceptLables):
# Function generates a heatmap of correlation
#
# Input arguments:
#   - dataframe dataExceptLables: panda's datafram of all attributes except lables 
# Output:
#   - Heatmap of correlation data
#   
#######################################################################################################################

def correlation_map(dataExceptLables):
    # Hint: import seaborn as sns and use sns.heatmap()
    corr = dataExceptLables.corr()
    ax = sns.heatmap(corr, xticklabels=True, yticklabels=True, vmin=-1, vmax=1, center=0,
                     cmap=sns.diverging_palette(20, 250, n=400), square=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right');
    plt.show()


#######################################################################################################################
# Function find_noof_clusters(dataExceptLables):
# Function generates a plot  to visualize the no of cluster of wdbc data
# Hint: use K-mean to find out no. of clusters in wdbc data
#
# Input arguments:
#   - dataframe dataExceptLables: panda's datafram of all attributes except lables 
# Output:
#   - a plot to visualize the no of clusters.
#   
#######################################################################################################################       
def find_noof_clusters(dataExceptLables):
    wcss = []  # within cluster sum of squares

    K = range(1, 15)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(dataExceptLables)
        # Compute the average of the squared distances from the cluster centers of the respective clusters using the Euclidean distance metric
        wcss.append(sum(np.min(cdist(dataExceptLables, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) /
                    dataExceptLables.shape[0])

    plt.plot(range(1, 15), wcss)
    plt.xlabel("K value")
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    plt.ylabel("WCSS")
    plt.show()


#######################################################################################################################
# Function compute_results(dataWithoutDiagnosis,data):
# Function computes the no. of correct and incorrect data items after k-mean clustering  by comparing the clusterd data lables with  origional lables
#
# Input arguments:
#   - dataframe dataWithoutDiagnosis: panda's datafram of 'radius_mean','fractal_dimension_mean' attributes
#   - dataframe data :  panda's datafram to get the origional lables
# Output:
#   - print Total in-correct
#   - print Total correct
#   -  % correct
#   
#######################################################################################################################  
def compute_results(dataWithoutDiagnosis, data):
    scalar = StandardScaler()
    kmeans = KMeans(n_clusters=2)

    scalar_data = scalar.fit_transform(dataWithoutDiagnosis)
    kmeans.fit(scalar_data)
    # print(kmeans.labels_)
    diagonsis = pd.Series(data['diagnosis'].replace(['M', "B"], [1, 0]))
    check_vals = array(diagonsis)
    # print(check_vals)
    correct = 0
    incorrect = 0
    total = len(kmeans.labels_)
    for i in range(total):
        if kmeans.labels_[i] == check_vals[i]:
            correct += 1
        else:
            incorrect += 1

    print("Result from kmeans with two attributes")

    print('Total in-correct = ', incorrect)
    print('Total correct = ', correct)
    print("% correct = ", correct / total)

    print("")


#######################################################################################################################
# Function compute_results_with_all_atributes(dataWithoutDiagnosis,data):
# Function computes the no. of correct and incorrect data items after k-mean clustering  using all attribues
#
# Input arguments:
#   - dataframe dataWithoutDiagnosis: panda's datafram of all attributes except lables
#   - dataframe data :  panda's datafram to get the origional lables
# Output:
#   - print Total in-correct
#   - print Total correct
#   -  % correct
#   
#######################################################################################################################      
def compute_results_with_all_atributes(dataWithoutDiagnosis, data):
    scalar = StandardScaler()
    kmeans = KMeans(n_clusters=2)

    scalar_data = scalar.fit_transform(dataWithoutDiagnosis)
    kmeans.fit(scalar_data)
    # print(kmeans.labels_)
    diagonsis = pd.Series(data['diagnosis'].replace(['M', "B"], [0, 1]))
    check_vals = array(diagonsis)
    # print(check_vals)
    correct = 0
    incorrect = 0
    total = len(kmeans.labels_)
    for i in range(total):
        if kmeans.labels_[i] == check_vals[i]:
            correct += 1
        else:
            incorrect += 1

    print("Result from kmeans with all attributes")

    print('Total in-correct = ', incorrect)
    print('Total correct = ', correct)
    print("% correct = ", correct / total)

    print("")


#######################################################################################################################
# Function compute_results_with_pca(dataWithoutDiagnosis,data):
# Function computes the no. of correct and incorrect data items after k-mean clustering  using all attribues
#
# Input arguments:
#   - dataframe dataWithoutDiagnosis: panda's datafram of all attributes except lables
#   - dataframe data :  panda's datafram to get the origional lables
# Output:
#   - print Total in-correct
#   - print Total correct
#   -  % correct
#   
#######################################################################################################################

def compute_results_with_pca(dataWithoutDiagnosis, data):
    # TODO:
    pca = PCA(20)

    scalar = StandardScaler()
    kmeans = KMeans(n_clusters=2)

    pca_data = pca.fit_transform(dataWithoutDiagnosis)
    scale_data = scalar.fit_transform(pca_data)
    kmeans.fit(scale_data)
    diagnosis = pd.Series(data['diagnosis'].replace(['M', "B"], [1, 0]))
    check_vals = array(diagnosis)
    correct = 0
    incorrect = 0
    total = len(kmeans.labels_)
    for i in range(total):
        if kmeans.labels_[i] == check_vals[i]:
            correct += 1
        else:
            incorrect += 1

    print("Result from kmeans with all attributes and using pca")

    print('Total in-correct = ', incorrect)
    print('Total correct = ', correct)
    print("% correct = ", correct / total)

    print("")


#######################################################################################################################
# Function Dendrogram(dataExceptLables):
# Function plot's the  hierarchy of wdbc data  using dendrogram
#
# Input arguments:
#   - dataframe dataExceptLables: panda's datafram of all attributes except lables
#   
# Output:
#   - dendrogram of provided dataframe 
# Hint: use the dendrogram,linkage from scipy.cluster.hierarchy  
####################################################################################################################### 

def Dendrogram(dataExceptLables):
    # Hint: use the dendrogram,linkage from scipy.cluster.hierarchy
    # TODO:

    # - compute the linkage
    # - plot the dendrogram of dataExceptLables
    link = linkage(dataExceptLables)
    dendrogram(link)
    plt.xlabel("data points")
    plt.ylabel("euclidean distance")
    plt.show()


#######################################################################################################################
# Function Agglomerative_clustering(dataExceptLables):
# Function perform the AgglomerativeClustering on provided data
#
# Input arguments:
#   - dataframe dataExceptLables: panda's datafram of all attributes except lables
#   
# Output:
#   - shows a plot of radius_mean and fractal_dimension_mean after AgglomerativeClustering 
#   - shows a plot of radius_mean and fractal_dimension_mean to compare data of before and after clustering
# Hint: use the AgglomerativeClustering from from sklearn.cluster 
####################################################################################################################### 
def Agglomerative_clustering(dataExceptLables):
    # Hint: import AgglomerativeClustering from sklearn.cluster

    # - perform hierarchal clustering  using AgglomerativeClustering algorithm
    aggloClust = AgglomerativeClustering()
    aggloClust.fit(dataExceptLables)
    labels = aggloClust.labels_

    # Data after hierarchical clustering
    plt.scatter(dataExceptLables["radius_mean"][labels == 0],
                dataExceptLables["fractal_dimension_mean"][labels == 0], color="red")
    plt.scatter(dataExceptLables["radius_mean"][labels == 1],
                dataExceptLables["fractal_dimension_mean"][labels == 1], color="blue")
    plt.xlabel("radius_mean")
    plt.ylabel("fractal_dimension_mean")
    plt.show()

    # Our data looks like below plot without diagnosis label

    plt.scatter(dataExceptLables["radius_mean"], dataExceptLables["fractal_dimension_mean"])
    plt.xlabel('radius_mean')
    plt.ylabel('fractal_dimension_mean')
    plt.show()


#######################################################################################################################
# Function Compute_results(dataExceptLables,data):
# Function computes the no. of correct and incorrect data items after AgglomerativeClustering clustering  using all attribues
#
# Input arguments:
#   - dataframe dataExceptLables: panda's datafram of all attributes except lables
#   - dataframe data :  panda's datafram to get the origional lables
# Output:
#   - print Total in-correct
#   - print Total correct
#   -  % correct
#   
#######################################################################################################################

def Compute_results(dataExceptLables, data):
    scalar = StandardScaler()
    dataExceptLables = scalar.fit_transform(dataExceptLables)
    # Hint: use AgglomerativeClustering from sklearn.cluster
    agg = AgglomerativeClustering()
    agg.fit(dataExceptLables)

    diagnosis = pd.Series(data['diagnosis'].replace(['M', "B"], [0, 1]))
    check_vals = array(diagnosis)
    correct = 0
    incorrect = 0
    total = len(agg.labels_)
    for i in range(total):
        if agg.labels_[i] == check_vals[i]:
            correct += 1
        else:
            incorrect += 1

    print("Result from agglomerative Clustering with all attributes:")

    print('Total in-correct = ', incorrect)
    print('Total correct = ', correct)
    print("% correct = ", correct / total)


if __name__ == '__main__':
    # Load the dataset
    data = pd.read_csv(bc_data_path, names=bc_data_columns).fillna(0.0)

    # Drop the  "id" as we don't need 
    data.drop(["id"], axis=1, inplace=True)
    # print(data.head())

    # count the lable values
    data["diagnosis"].value_counts()
    print(data["diagnosis"].value_counts())

    # For clustering analysis we do not need labels. As we will identify the labels. So drop the diagnosis attribute

    dataExceptLables = data.drop(["diagnosis"], axis=1)
    dataExceptLables.head()

    print(dataExceptLables.info())
    correlation_map(dataExceptLables)

    find_noof_clusters(dataExceptLables)

    # #compute the results with radius_mean and fractal_dimension_mean attributes

    dataWithoutDiagnosis = data.loc[:, ['radius_mean', 'fractal_dimension_mean']]
    compute_results(dataWithoutDiagnosis, data)

    # # #compute the results with all attribues    

    dataWithoutDiagnosis = data.drop(["diagnosis"], axis=1)
    compute_results_with_all_atributes(dataWithoutDiagnosis, data)

    # # # #compute the results with all attribues using PCA  
    dataWithoutDiagnosis = data.drop(["diagnosis"], axis=1)
    compute_results_with_pca(dataWithoutDiagnosis, data)

    # plot the dendrogram of dataExceptLables
    Dendrogram(dataExceptLables)

    Agglomerative_clustering(dataExceptLables)

    Compute_results(dataExceptLables,data)
