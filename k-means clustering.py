import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('D:\\cognitior\Basics of data science\\dataset\\Mall_Customers.csv')

x = dataset.iloc[:, [3,4]].values
from sklearn.cluster import KMeans

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++')
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
#IT SHOW THW ELBOW LINE FOR THE NUMBER OF CLUSTERS
plt.plot(range(1,11), wcss)

kmeans = KMeans(n_clusters=5, init='k-means++',random_state=0)
y_kmeans = kmeans.fit_predict(x)

pd.concat([dataset, pd.DataFrame(y_kmeans)], axis=1)
#for ploting the points ion clusters
plt.scatter(x[y_kmeans==0,0], x[y_kmeans==0,1], s=100, c='red', label='cluster1',)
plt.scatter(x[y_kmeans==1,0], x[y_kmeans==1,1], s=100, c='pink', label='cluster1')
plt.scatter(x[y_kmeans==2,0], x[y_kmeans==2,1], s=100, c='yellow', label='cluster1')
plt.scatter(x[y_kmeans==3,0], x[y_kmeans==3,1], s=100, c='black', label='cluster1')
plt.scatter(x[y_kmeans==4,0], x[y_kmeans==4,1], s=100, c='green', label='cluster1')

