
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


# Load the irsi dataset to variable
iris=datasets.load_iris()

resultDataFrame = pd.DataFrame(columns=['Cluster', 'Metrics Accuracy', 'NMIS Accuracy'])

df = pd.DataFrame(iris.data)
df.columns=iris.feature_names
df['target']=iris.target

# iris data
X=iris.data
print('iris data:', X)
# iris features
print('iris features:', iris.feature_names)

# iris target 
Y=iris.target
print('iris target:', Y)

# printing target names
print('iris features:', iris.target_names)

# printing the shapes (data and target)
print('iris data shape', iris.data.shape)
print('iris target shape', iris.target.shape)


# --------------------------------Feautre Selection----------------------------

# --------------- t-SNE to feature selection 

tsne = manifold.TSNE(n_components=2, init='pca')
X_tsn = tsne.fit_transform(X)
print('tsne shape of the data', X_tsn.shape)


# --------------------------Clustering Methods --------------------------------
# 1- Kmeans Clustering --------------------------------------------------------

KMeans_x = cluster.KMeans(n_clusters=len(iris.target_names))
# fitting
KMeans_x.fit(X_tsn,Y)
#prediction
Y_pred = KMeans_x.predict(X_tsn)

# data frame
df = pd.DataFrame(X_tsn)

df['target'] = Y
df['Y_pred'] = Y_pred


# Visualize Clustered data VS Feature selected data from raw 
plt.figure(figsize=(9, 8))
plt.subplot(1, 2, 1)
plt.title("Feature Selection of raw data")
ind = Y==0
plt.scatter(X_tsn[ind,0],X_tsn[ind,1],c='red',marker='*')
ind = Y==1
plt.scatter(X_tsn[ind,0],X_tsn[ind,1],c='cyan',marker='*')
ind = Y==2
plt.scatter(X_tsn[ind,0],X_tsn[ind,1],c='black',marker='*')

plt.subplot(1, 2, 2)
plt.title("Kmeans Clustered Data")
ind = KMeans_x.labels_==0
plt.scatter(X_tsn[ind,0],X_tsn[ind,1],c='red',marker='o')
ind = KMeans_x.labels_==1
plt.scatter(X_tsn[ind,0],X_tsn[ind,1],c='blue',marker='o')
ind = KMeans_x.labels_==2
plt.scatter(X_tsn[ind,0],X_tsn[ind,1],c='yellow',marker='o')

plt.show()

# printing the erorrs
matrix_ac_score = metrics.accuracy_score(Y, Y_pred)
NMIS_km = normalized_mutual_info_score(Y, Y_pred)

#print('Matrix accuracy score for Kmeans Clustering is: {0:.2f}'.format(matrix_ac_score))
print('NMI score for Kmeans Clustering is: {0:.2f}'.format(NMIS_km))

#pdb.set_trace()
#
resultDataFrame.loc[0]=['Kmeans clustering', NMIS_km, matrix_ac_score]
print('Data frame Result for Kmeans clustering', resultDataFrame.loc[0])

#
# 2- Gaussian Mixture clustering ----------------------------------------------
#

gmmcl = mixture.GMM(n_components=3, covariance_type='full')
gmmcl.fit(X_tsn,Y)


Y_pred = gmmcl.predict(X_tsn)

# Visualize Clustered data VS Feature selected data from raw
plt.figure(figsize=(9, 8))
plt.subplot(1, 2, 1)
plt.title("Feature Selection of raw data")
ind = Y==0
plt.scatter(X_tsn[ind,0],X_tsn[ind,1],c='red',marker='*')
ind = Y==1
plt.scatter(X_tsn[ind,0],X_tsn[ind,1],c='green',marker='*')
ind = Y==2
plt.scatter(X_tsn[ind,0],X_tsn[ind,1],c='cyan',marker='*')

plt.subplot(1, 2, 2)
plt.title("Gaussian Mixture Clustered Data")
ind = Y_pred==0
plt.scatter(X_tsn[ind,0],X_tsn[ind,1],c='red',marker='<')
ind = Y_pred==1
plt.scatter(X_tsn[ind,0],X_tsn[ind,1],c='green',marker='<')
ind = Y_pred==2
plt.scatter(X_tsn[ind,0],X_tsn[ind,1],c='cyan',marker='<')

plt.show()

# printing the erorrs
gmmcl_mat_acc_acr = metrics.accuracy_score(Y, Y_pred)
NMIS_gm = normalized_mutual_info_score(Y, Y_pred)

#print('Matrix accuracy score for Gaussian Mixture Clustering is: {0:.2f}'.format(matrix_ac_score))
print('NMI score for Gaussian Mixture Clustering is: {0:.2f}'.format(NMIS_gm))

#
resultDataFrame.loc[1]=['Gaussian Mixture clustering', NMIS_gm, gmmcl_mat_acc_acr]
print('Data frame Result for Gaussian Mixture clustering', resultDataFrame.loc[1])

#
# 3- AffinityPropagation Clustering---------------------------------------------
#

affpr = cluster.AffinityPropagation()
affpr.fit(X_tsn,Y)

# prediction based on features
Y_pred = affpr.predict(X_tsn)

# Visualize Clustered data VS Feature selected data from raw
plt.figure(figsize=(9, 8))
plt.subplot(1, 2, 1)
plt.title("Feature Selection of raw data")
ind = Y==0
plt.scatter(X_tsn[ind,0],X_tsn[ind,1],c='red',marker='*')
ind = Y==1
plt.scatter(X_tsn[ind,0],X_tsn[ind,1],c='green',marker='*')
ind = Y==2
plt.scatter(X_tsn[ind,0],X_tsn[ind,1],c='cyan',marker='*')

plt.subplot(1, 2, 2)
plt.title("Affinity Propagation Clustered Data")
ind = Y_pred==0
plt.scatter(X_tsn[ind,0],X_tsn[ind,1],c='red',marker='+')
ind = Y_pred==1
plt.scatter(X_tsn[ind,0],X_tsn[ind,1],c='green',marker='+')
ind = Y_pred==2
plt.scatter(X_tsn[ind,0],X_tsn[ind,1],c='cyan',marker='+')

plt.show()
# 

# printing the erorrs
affpr_mat_acc_scr = metrics.accuracy_score(Y, Y_pred)
NMIS_ap = normalized_mutual_info_score(Y, Y_pred)

#print('Matrix accuracy score for Affinity Propagation Clustering is: {0:.2f}'.format(matrix_ac_score))
print('NMI score for Affinity Propagation Clustering is: {0:.2f}'.format(NMIS_ap))

#
resultDataFrame.loc[2]=['Affinity Propagation clustering', NMIS_ap, affpr_mat_acc_scr]
print('Data frame Result for Affinity Propagation clustering', resultDataFrame.loc[2])

#
# 4- Birch clustering -------------------------------------------------------
#

birch = cluster.Birch()
birch.fit(X_tsn, Y)


Y_pred = birch.predict(X_tsn)


# Visualize Clustered data VS original Feature selected data
plt.figure(figsize=(9, 8))
plt.subplot(1, 2, 1)
plt.title("Feature Selection of raw data")
ind = Y==0
plt.scatter(X_tsn[ind,0],X_tsn[ind,1],c='red',marker='*')
ind = Y==1
plt.scatter(X_tsn[ind,0],X_tsn[ind,1],c='green',marker='*')
ind = Y==2
plt.scatter(X_tsn[ind,0],X_tsn[ind,1],c='cyan',marker='*')

plt.subplot(1, 2, 2)
plt.title("Birch Clustered Data")
ind = Y_pred==0
plt.scatter(X_tsn[ind,0],X_tsn[ind,1],c='red',marker='^')
ind = Y_pred==1
plt.scatter(X_tsn[ind,0],X_tsn[ind,1],c='green',marker='^')
ind = Y_pred==2
plt.scatter(X_tsn[ind,0],X_tsn[ind,1],c='cyan',marker='^')

plt.show()


# printing the erorrs
brch_mat_acc_acr = metrics.accuracy_score(Y, Y_pred)
NMIS_br = normalized_mutual_info_score(Y, Y_pred)

#print('Matrix accuracy score for Birch  Clustering is: {0:.2f}'.format(matrix_ac_score))
print('NMI score for Birch  Clustering is: {0:.2f}'.format(NMIS_br))

#
resultDataFrame.loc[3]=['Birch  clustering', NMIS_br, brch_mat_acc_acr]
print('Data frame Result for Birch  clustering', resultDataFrame.loc[3])


#
# 5- Agglomerative Clustering clustering ---------------------------------------
#

aggcl = cluster.AgglomerativeClustering()
aggcl.fit(X_tsn,Y)

#
Y_pred = aggcl.labels_
#

# Visualize Clustered data VS Feature selected data from raw
plt.figure(figsize=(9, 8))
plt.subplot(1, 2, 1)
plt.title("Feature Selection of raw data")
ind = Y==0
plt.scatter(X_tsn[ind,0],X_tsn[ind,1],c='red',marker='*')
ind = Y==1
plt.scatter(X_tsn[ind,0],X_tsn[ind,1],c='green',marker='*')
ind = Y==2
plt.scatter(X_tsn[ind,0],X_tsn[ind,1],c='cyan',marker='*')

plt.subplot(1, 2, 2)
plt.title("Agglomorative Clustered Data")
ind = Y_pred==0
plt.scatter(X_tsn[ind,0],X_tsn[ind,1],c='red',marker='>')
ind = Y_pred==1
plt.scatter(X_tsn[ind,0],X_tsn[ind,1],c='green',marker='>')
ind = Y_pred==2
plt.scatter(X_tsn[ind,0],X_tsn[ind,1],c='cyan',marker='>')

plt.show()

# printing the erorrs
aggcl_mat_acc_acr = metrics.accuracy_score(Y, Y_pred)
NMIS_ag = normalized_mutual_info_score(Y, Y_pred)

#print('Matrix accuracy score for Agglomerative Clustering is: {0:.2f}'.format(matrix_ac_score))
print('NMI score for Agglomerative Clustering is: {0:.2f}'.format(NMIS_ag))

#
resultDataFrame.loc[4]=['Agglomerative clustering', NMIS_ag, aggcl_mat_acc_acr]
print('Data frame Result for Agglomerative clustering', resultDataFrame.loc[4])



# resulted data frame for all clusterings
print('Data frame Result for 5 clustering', resultDataFrame) 