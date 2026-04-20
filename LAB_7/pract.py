from sklearn.model_selection inport train_test_split

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
# pca using svd + co-varinace matrix

pca = PCA(n_components=2) # no need to emntion always

Z = pca.fit_transform(X)

# Z-contains tarnformed data ...

var_ratios = pca.explained_variance_ratio_
# instaed of n_components hard coded::
# cum_sum = np.cumsum(pca.explained_variance_ratio_)
#n_comp_opt = np.argmax(cum_sum >= threshold) + 1

plt.scatter(Z[:,0],Z[:,1],c=y)
plt.xlabel("FIRST COMPONENTS")
plt.ylabel("Y-COMPONENTS")
plt.title("ANALYSIS")
plt.show()

kmean = KMeans(n_clusters=3,random_state = 42)

labels = kmean.fit_predict(X) # fit - enugh bit dorect ehre gives assigned cliusters..

centriods = kmean.cluster_centers_

wcss = kmean.inertia_

plt.scatter(x[:,0],x[:,1],c=labels)

plt.scatter(centriods[:,0],centrios[:,1],c='red')
plt.xlabel("ijnc")
plt.ylabel("bci")
plt.title("ncie")
plt.show()

wcss=[]

for i in range(1,7):
    kmeans = KMeans(n_clusters = i,random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,7),wcss,marker='o')
plt.xlabel("---")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# # #  ##  #               #### # #  # ##  ##  # ## 

pca.fit(X) # when n_components not fixed..

pca.transform(X_test) ##

np.linalg.norm(X-Xapp)

Xapp = pca.inverse_tarnsform(Z)

