
# coding: utf-8

# Ali Nehrani

import numpy as np
import pandas as pd
from sklearn import  (datasets, metrics,  cluster, feature_selection, manifold,  
decomposition, preprocessing, mixture)
from matplotlib import pyplot as plt
from IPython.core.debugger import Tracer
from sklearn.metrics import normalized_mutual_info_score

import pdb



# Load the boston dataset to variable
boston = datasets.load_boston()

resultDataFrame = pd.DataFrame(columns=['Cluster',  'NMIS Accuracy'])

df = pd.DataFrame(boston.data)
df.columns=boston.feature_names
df['target']=boston.target

# boston data
X=boston.data
#print('boston data shape:', boston.data.shape)
print('boston data:', X)
# iris features
print('boston features:', boston.feature_names)
#print('boston target head', boston.data.head)
# boston target 
Y=boston.target
print('boston target:', Y)

# printing target names
#print('boston features:', boston.target_names)

# printing the shapes (data and target)
print('boston data shape', boston.data.shape)
print('boston target shape', boston.target.shape)

#pandas representation of the boston data
pboston = pd.DataFrame(boston.data)
pboston.columns = boston.feature_names
print('boston data with pandas', pboston.head())

# --------------------------------Feautre Selection----------------------------

# --------------- t-SNE to feature selection 

tsne = manifold.TSNE(n_components=2, init='pca')
X_tsn = tsne.fit_transform(X)
print('tsne shape of the boston data', X_tsn.shape)


# --------------------------Clustering Methods --------------------------------
# 1- Kmeans Clustering --------------------------------------------------------
#c


Pair_bs = np.array([np.concatenate((X_tsn[i],[Y[i]])) for i in range(len(boston.data))])


# number of clusters is optional - I set it to 7
n_clusters = 7 # choose less that 13



KMeans_x = cluster.KMeans(n_clusters, max_iter=500, verbose=1)
# fitting
#KMeans_x.fit(X_tsn,Y)
KMeans_x.fit(Pair_bs)
#prediction
#labels = KMeans_x.labels_
#centroids = KMeans_x.cluster_centers_
#Y = labels
#Y_pred = KMeans_x.predict(X_tsn)

Y_pred = KMeans_x.predict(Pair_bs)


centroids = KMeans_x.cluster_centers_


# data frame
df = pd.DataFrame(X_tsn)


#df['target'] = KMeans_x.cluster_centers_
df['Y_pred'] = Y_pred

# representing with pandas
#pd.DataFrame(boston.data, columns=boston.feature_names)



# Visualize Clustered data VS Feature selected data from raw 
plt.figure(figsize=(9, 8))
plt.subplot(1, 2, 1)
plt.title("clustering data with Kmean")

color_lst = ['red', 'cyan', 'gray', 'pink', 'purple', 'yellow', 'blue', 'green', 'orange', 'brown', 'olive', 'indigo', 'violet']

colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(set(Y_pred)))]

for i in range(n_clusters):
#    datapoints = X_tsn[np.where(Y_pred==i)]
    plt.scatter(Pair_bs[np.where(Y_pred==i),0], Pair_bs[np.where(Y_pred==i),1], c=colors[i], marker='*')
    
    centers = plt.plot(centroids[i,0],centroids[i,1],'x')   
    # Plot the centroids.
    # plt.setp(data_cl, markersize=1.0)
    plt.setp(centers,markersize=10.0)
    plt.setp(centers,markeredgewidth=5.0)

plt.xlim([-220,20])
plt.ylim([-40,60])
plt.show()



# printing the erorrs
#matrix_ac_score = metrics.accuracy_score(Y, Y_pred)
NMIS_km = normalized_mutual_info_score(Y, Y_pred)

#print('Matrix accuracy score for Kmeans Clustering is: {0:.2f}'.format(matrix_ac_score))
print('NMI score for Kmeans Clustering is: {0:.2f}'.format(NMIS_km))


#
resultDataFrame.loc[0]=['Kmeans clustering', NMIS_km]
print('Data frame Result for Kmeans clustering', resultDataFrame.loc[0])

#
# 2- Gaussian Mixture clustering ----------------------------------------------
#

 

gmmcl = mixture.GaussianMixture(n_components=n_clusters, covariance_type='full')
gmmcl.fit(Pair_bs)


Y_pred = gmmcl.predict(Pair_bs)

#centroids = gmmcl.cluster_centers_




# Visualize Clustered data VS Feature selected data from raw
plt.figure(figsize=(9, 8))
plt.subplot(1, 2, 1)
plt.title("clustering data with Gaussian Mixture")

color_lst = ['red', 'cyan', 'gray', 'pink', 'purple', 'yellow', 'blue', 'green', 'orange', 'brown', 'olive', 'indigo', 'violet']

colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(set(Y_pred)))]

for i in range(n_clusters):
#    datapoints = X_tsn[np.where(Y_pred==i)]
    mainplot = plt.scatter(Pair_bs[np.where(Y_pred==i),0], Pair_bs[np.where(Y_pred==i),1], c=colors[i], marker='>')
    
#    centers = plt.plot(centroids[i,0],centroids[i,1],'x')   
    # Plot the centroids.
    # plt.setp(data_cl, markersize=1.0)
#    plt.setp(centers,markersize=10.0)
#    plt.setp(centers,markeredgewidth=5.0)

plt.xlim([-220,20])
plt.ylim([-40,40])
plt.show()




# printing the erorrs
#gmmcl_mat_acc_acr = metrics.accuracy_score(Y, Y_pred)
NMIS_gm = normalized_mutual_info_score(Y, Y_pred)

#print('Matrix accuracy score for Gaussian Mixture Clustering is: {0:.2f}'.format(matrix_ac_score))
print('NMI score for Gaussian Mixture Clustering is: {0:.2f}'.format(NMIS_gm))

#
resultDataFrame.loc[1]=['Gaussian Mixture clustering', NMIS_gm] #, gmmcl_mat_acc_acr
print('Data frame Result for Gaussian Mixture clustering', resultDataFrame.loc[1])

#
# 3- Affinity Propagation Clustering---------------------------------------------
#



affpr = cluster.AffinityPropagation()
affpr.fit(Pair_bs)

# prediction based on features
Y_pred = affpr.predict(Pair_bs)

centroids = affpr.cluster_centers_

# Visualize Clustered data VS Feature selected data from raw
plt.figure(figsize=(9, 8))
plt.subplot(1, 2, 1)
plt.title("clustering data with Affinity Propagation")

color_lst = ['red', 'cyan', 'gray', 'pink', 'purple', 'yellow', 'blue', 'green', 'orange', 'brown', 'olive', 'indigo', 'violet']

colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(set(Y_pred)))]

for i in range(np.max(Y_pred)):
#    datapoints = X_tsn[np.where(Y_pred==i)]
    mainplot = plt.scatter(Pair_bs[np.where(Y_pred==i),0], Pair_bs[np.where(Y_pred==i),1], c=colors[i], marker='>')
    
    centers = plt.plot(centroids[i,0],centroids[i,1],'x')   
    # Plot the centroids.
    # plt.setp(data_cl, markersize=1.0)
    plt.setp(centers,markersize=3.0)
    plt.setp(centers,markeredgewidth=2.0)

plt.xlim([-220,20])
plt.ylim([-40,60])
plt.show()
# 

# printing the erorrs
#affpr_mat_acc_scr = metrics.accuracy_score(Y, Y_pred)
NMIS_ap = normalized_mutual_info_score(Y, Y_pred)

#print('Matrix accuracy score for Affinity Propagation Clustering is: {0:.2f}'.format(matrix_ac_score))
print('NMI score for Affinity Propagation Clustering is: {0:.2f}'.format(NMIS_ap))

#
resultDataFrame.loc[2]=['Affinity Propagation clustering', NMIS_ap]
print('Data frame Result for Affinity Propagation clustering', resultDataFrame.loc[2])



#
# 4- Birch clustering -------------------------------------------------------
#


Birch = cluster.Birch(n_clusters=n_clusters)
Birch.fit(Pair_bs)


Y_pred = Birch.predict(Pair_bs)

centroids = Birch.subcluster_centers_ 
# Visualize Clustered data VS Feature selected data from raw
plt.figure(figsize=(9, 8))
plt.subplot(1, 2, 1)
plt.title("clustering data with Birch Clustering")

color_lst = ['red', 'cyan', 'gray', 'pink', 'purple', 'yellow', 'blue', 'green', 'orange', 'brown', 'olive', 'indigo', 'violet']

colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(set(Y_pred)))]

for i in range(np.max(Y_pred)):
#    datapoints = X_tsn[np.where(Y_pred==i)]
    mainplot = plt.scatter(Pair_bs[np.where(Y_pred==i),0], Pair_bs[np.where(Y_pred==i),1], c=colors[i], marker='>')
    
    centers = plt.plot(centroids[i,0],centroids[i,1],'x')   
    # Plot the centroids.
    # plt.setp(data_cl, markersize=1.0)
    plt.setp(centers,markersize=5.0)
    plt.setp(centers,markeredgewidth=2.0)

plt.xlim([-220,20])
plt.ylim([-40,60])
plt.show()
# 


# printing the erorrs
#brch_mat_acc_acr = metrics.accuracy_score(Y, Y_pred)
NMIS_br = normalized_mutual_info_score(Y, Y_pred)

#print('Matrix accuracy score for Birch  Clustering is: {0:.2f}'.format(matrix_ac_score))
print('NMI score for Birch  Clustering is: {0:.2f}'.format(NMIS_br))

#
resultDataFrame.loc[3]=['Birch  clustering', NMIS_br]
print('Data frame Result for Birch  clustering', resultDataFrame.loc[3])



#
# 5- Mean Shift Clustering  ---------------------------------------
#

MeanS = cluster.MeanShift(n_jobs=n_clusters)
MeanS.fit(Pair_bs)

centroids = MeanS.cluster_centers_

#
Y_pred = MeanS.labels_
#


# Visualize Clustered data VS Feature selected data from raw
plt.figure(figsize=(9, 8))
plt.subplot(1, 2, 1)
plt.title("clustering data with Mean Shift Clustering")

color_lst = ['red', 'cyan', 'gray', 'pink', 'purple', 'yellow', 'blue', 'green', 'orange', 'brown', 'olive', 'indigo', 'violet']
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(set(Y_pred)))]

for i, col in zip(Y_pred, colors):

    plt.plot(Pair_bs[np.where(Y_pred==i), 0], Pair_bs[np.where(Y_pred==i), 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=4)
    centers = plt.plot(centroids[i,0],centroids[i,1],'x')   
    # Plot the centroids.
    # plt.setp(data_cl, markersize=1.0)
    plt.setp(centers,markersize=5.0)
    plt.setp(centers,markeredgewidth=2.0)

plt.xlim([-220,20])
plt.ylim([-40,60])
plt.show()
# 

# printing the erorrs
#aggcl_mat_acc_acr = metrics.accuracy_score(Y, Y_pred)
NMIS_ag = normalized_mutual_info_score(Y, Y_pred)

#print('Matrix accuracy score for Agglomerative Clustering is: {0:.2f}'.format(matrix_ac_score))
print('NMI score for Mean Shift is: {0:.2f}'.format(NMIS_ag))

#
resultDataFrame.loc[4]=['Mean Shift clustering', NMIS_ag]
print('Data frame Result for Mean Shift clustering', resultDataFrame.loc[4])



# resulted data frame for all clusterings
print('Data frame Result for 5 clustering', resultDataFrame) 



#
# 6- Agglomerative Clustering clustering ---------------------------------------
#

aggcl = cluster.AgglomerativeClustering(n_clusters = n_clusters)
aggcl.fit(Pair_bs)

#
Y_pred = aggcl.labels_
#
# printing the erorrs
#aggcl_mat_acc_acr = metrics.accuracy_score(Y, Y_pred)
NMIS_ag = normalized_mutual_info_score(Y, Y_pred)

#print('Matrix accuracy score for Agglomerative Clustering is: {0:.2f}'.format(matrix_ac_score))
print('NMI score for Agglomerative Clustering is: {0:.2f}'.format(NMIS_ag))

#
resultDataFrame.loc[5]=['Agglomerative clustering', NMIS_ag]
print('Data frame Result for Agglomerative clustering', resultDataFrame.loc[5])



# resulted data frame for all clusterings
print('Data frame Result for 5 clustering', resultDataFrame) 
