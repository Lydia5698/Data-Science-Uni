import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import array


# For reproducibility
np.random.seed(1000)

bc_data_path = 'wdbc.data' # path to the input data file.

bc_data_columns = ['id','diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean','area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
                      'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean','radius_se','texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
                      'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se','fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst',
                      'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst','concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']



#######################################################################################################################
# Function db_scan(dataExceptLables, labels):
# Function perform the Density-based spatial clustering on provided data
#
# Input arguments:
#   - dataframe dataExceptLables: panda's datafram of attributes except lables
#   - Series of Lables: panda's datafram series of lables
#   
# Output:
#   - no. of estimated clusters using DBSCAN
#   - no. of noise points using DBSCAN
#   - remove the noise points from data and labels
#   - shows the plot of data with noise points
#   - shows the plot of data with out noise points
# Hint: use DBSCAN  from the sklearn.cluster 
####################################################################################################################### 
def db_scan(dataExceptLables,labels_true):

    

    X = StandardScaler().fit_transform(dataExceptLables)
    pca = PCA(20)
    pca_data = pca.fit_transform(X)
    #TODO:
    #  Implement the DBSCAN(eps=0.3, min_samples=10)
    #  estimate the number of clusters
    #  estimate the noise pints
    #  Print the Estimated number of clusters
    #  Print the Estimated number of noise points
    # return X: Array of shape [n_samples, n_features]. (Feature Matrix)
    #  Show the graph  with number of cluster and noise points
    #  Show the graph  with-out noise points

    db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    labels = db.labels_
    clusters = array.unique(labels)

    if -1 in clusters:
        number_classes = len(clusters)-1
    else:
        number_classes = len(clusters)

    noise_points = 0
    for i in labels:
        if i == -1:
            noise_points += 1



    print('Estimated number of clusters:')
    print('Estimated number of noise points:')
    plt.figure()
    plt.scatter(dataExceptLables["radium_mean"][labels == 0], dataExceptLables["fractial_dimension"])
   





    plt.show()


#######################################################################################################################
# Function db_scan_pca(dataExceptLables, labels):
# Function perform the Density-based spatial clustering on provided data after Principal component analysis (PCA )
#
# Input arguments:
#   - dataframe dataExceptLables: panda's datafram of attributes except lables
#   - Series of Lables: panda's datafram series of lables
#   
# Output:
#   - no. of estimated clusters using DBSCAN
#   - no. of noise points using DBSCAN
#   - remove the noise points from data and labels
#   - shows the plot of data with noise points
#   - shows the plot of data with out noise points
#   - compare the results of 
# Hint: use DBSCAN  from the sklearn.cluster db_scan() and db_scan_pca() functions
####################################################################################################################### 

def db_scan_pca(dataExceptLables,labels_true):

    #TODO:
    #  Perform the PCA on the provided data 
    #  Implement the DBSCAN(eps=0.3, min_samples=10)
    #  generate the random data using n_samples and center parameter values (hint: use make_blobs())
    #  estimate the number of clusters
    #  estimate the noise pints
    #  Print the Estimated number of clusters
    #  Print the Estimated number of noise points
    # return X: Array of shape [n_samples, n_features]. (Feature Matrix)
    #  Show the graph  with number of cluster and noise points
    #  Show the graph  with-out noise points
    # compare the results of both function

    
    

    print('Estimated number of clusters after PCA:')
    print('Estimated number of noise points after PCA:')
    
   



    plt.show()
    


if __name__ == '__main__':
   
    # Load the dataset
    data = pd.read_csv(bc_data_path, names=bc_data_columns).fillna(0.0)
    
    # Drop the  "id" as we don't need 
    data.drop(["id"], axis = 1, inplace = True)
        
    #count the lable values 
    data["diagnosis"].value_counts()
    print( data["diagnosis"].value_counts())
           
    
    dataExceptLables = data.loc[:,['radius_mean','fractal_dimension_mean']]
    diagonsis = pd.Series(data['diagnosis'].replace(['M',"B"],[0,1]))
    
    true_labels = array(diagonsis)
    dataExceptLables.head()
    
    print(dataExceptLables.info())
    db_scan(dataExceptLables,true_labels)
    db_scan_pca(dataExceptLables,true_labels)