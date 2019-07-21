import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('D:\\cognitior\Basics of data science\\dataset\\Mall_Customers.csv')

x = dataset.iloc[:, [3,4]].values
import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(x,method='ward'))
plt.title('Dendrogram')


from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', 
                             linkage='ward')

y_hc = hc.fit_predict(x)

pd.concat([dataset, pd.DataFrame(y_hc)], axis=1)

plt.scatter(x[y_hc==0,0], x[y_hc==0,1], s=100, c='red', label='cluster1')
plt.scatter(x[y_hc==1,0], x[y_hc==1,1], s=100, c='pink', label='cluster1')
plt.scatter(x[y_hc==2,0], x[y_hc==2,1], s=100, c='yellow', label='cluster1')
plt.scatter(x[y_hc==3,0], x[y_hc==3,1], s=100, c='black', label='cluster1')
plt.scatter(x[y_hc==4,0], x[y_hc==4,1], s=100, c='green', label='cluster1')
