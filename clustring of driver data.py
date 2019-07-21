import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#For k means clustring
dataset = pd.read_csv('D:\\cognitior\Basics of data science\\dataset\\driver-data.csv')

x = dataset.iloc[:,1:3].values

from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++')
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
#IT SHOW THW ELBOW LINE FOR THE NUMBER OF CLUSTERS
plt.plot(range(1,11), wcss)

kmeans = KMeans(n_clusters=2, init='k-means++',random_state=0)
y_kmeans = kmeans.fit_predict(x)

pd.concat([dataset, pd.DataFrame(y_kmeans)], axis=1)
#for ploting the points ion clusters
plt.scatter(x[y_kmeans==0,0], x[y_kmeans==0,1], s=100, c='red', label='cluster1')
plt.scatter(x[y_kmeans==1,0], x[y_kmeans==1,1], s=100, c='black', label='cluster1')


#For heirachical clustering
import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(x,method='ward'))
plt.title('Dendrogram')


from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=2, affinity='euclidean', 
                             linkage='ward')

y_hc = hc.fit_predict(x)

pd.concat([dataset, pd.DataFrame(y_hc)], axis=1)

plt.scatter(x[y_hc==0,0], x[y_hc==0,1], s=100, c='red', label='cluster1')
plt.scatter(x[y_hc==1,0], x[y_hc==1,1], s=100, c='pink', label='cluster1')
