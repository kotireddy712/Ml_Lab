# ================================
# IMPORTS & DATA LOADING
# ================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from collections import Counter

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# ================================
# PART A: PCA USING COVARIANCE
# ================================

# Step 1: Center data
mean = np.mean(X, axis=0)
Xc = X - mean

# Step 2: Covariance matrix
n = Xc.shape[0]
cov_matrix = (1/(n-1)) * np.dot(Xc.T, Xc)

# Step 3: Eigen decomposition
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

# Step 4: Sort eigenvalues and eigenvectors
idx = np.argsort(eig_vals)[::-1]
eig_vals = eig_vals[idx]
eig_vecs = eig_vecs[:, idx]

# Step 5: Reduce to 2D
W = eig_vecs[:, :2]
Z_cov = np.dot(Xc, W) # dataset converting to 2D..

# Plot PCA (Covariance)
plt.figure() # Creates a new blank canvas (graph window)
plt.scatter(Z_cov[:,0], Z_cov[:,1], c=y, cmap='viridis') # Plots points (dots) on graph
plt.title("PCA using Covariance")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid()
plt.show()

plt.figure() # optinal 
plt.scatter(X[:, 2], X[:, 3], c=y)
plt.xlabel("petal-length")
plt.ylabel("petal-width")
plt.title("original-plot")
plt.show()
# Explained variance
# Each eigenvalue 𝜆i --
# Represents variance along that principal component
# Explained variance ratio tells the proportion of total variance captured by each principal component, helping us decide how many components to retain.
#Before PCA, data is represented in original feature space (petal length and width),
#Before PCA, data is represented in original feature space (petal length and width),
explained_var = eig_vals / np.sum(eig_vals)
print("Explained Variance Ratio (Top 2):", explained_var[:2])

# ================================
# PART B: PCA USING SVD
# ================================
# Xc = U*(sigma)*(V)t
# U -- > n*d;
# Sigma -- > d*d
# V --> d*d
# sigma)i = SQRT(lambda)I

# V transpose conatins eigen-vectors..


# Step 1: Perform SVD
U, S, Vt = np.linalg.svd(Xc)

# Step 2: Reduce to 2D
V2 = Vt.T[:, :2]
Z_svd = np.dot(Xc, V2) # REDUCING trainingg-data DATA FROM 4D -- > 2D..

# Plot PCA (SVD)
plt.figure()
plt.scatter(Z_svd[:,0], Z_svd[:,1], c=y, cmap='viridis')
plt.title("PCA using SVD")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid()
plt.show()

# Step 3: Reconstruction (FIXED: add mean back)
X_approx = np.dot(Z_svd, V2.T) + mean

# Plot Original vs Reconstructed (Petal features)
plt.figure()
plt.scatter(X[:,2], X[:,3], label="Original", alpha=0.6)
plt.scatter(X_approx[:,2], X_approx[:,3], label="Reconstructed", alpha=0.6)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Original vs Reconstructed Data")
plt.legend()
plt.grid()
plt.show()

# Step 4: Reconstruction error (correct)
error = np.linalg.norm(X - X_approx, 'fro')
print("Reconstruction Error:", error)


# ================================
# PART C: K-MEANS FROM SCRATCH
# ================================

# Use only petal features
X_km = X[:, [2,3]]

# -------- Distance (VECTORISED) --------
# Compute distance of every point to every centroid
# each row says -- fro sample-i :: distances froma ll k-clusters distances 
def compute_distances(X, centroids):
    k = centroids.shape[0]
    distances = np.zeros((X.shape[0], k))
    for i in range(k):
        distances[:, i] = np.sum((X - centroids[i])**2, axis=1)
    return distances

# -------- Assign clusters --------
 # Compute distance of every point to every centroid
def assign_clusters(X, centroids):
    distances = compute_distances(X, centroids)
    return np.argmin(distances, axis=1)

# -------- Update centroids --------
 # Recompute centroids (mean of cluster points)
def update_centroids(X, labels, k):
    new_centroids = []
    for i in range(k):
        points = X[labels == i]
        if len(points) > 0:
            new_centroids.append(points.mean(axis=0)) # finding new-mean and appending..
        else:
            new_centroids.append(X[np.random.randint(len(X))])
    return np.array(new_centroids)

# -------- Compute WCSS --------
def compute_wcss(X, labels, centroids):
    wcss = 0
    for i in range(len(X)):
        c = centroids[labels[i]]
        wcss += np.sum((X[i] - c)**2)
    return wcss
# wcss == total clustering error 
# -------- K-Means Function --------
def kmeans(X, k, init_type="first", max_iters=20):

    # Initialization
    if init_type == "first":
        centroids = X[:k].copy()
    elif init_type == "random":
        idx = np.random.choice(len(X), k, replace=False)
        centroids = X[idx]

    print("\nInitial Centroids:\n", centroids)

    for it in range(max_iters):
        print(f"\nIteration {it+1}")
        labels = assign_clusters(X, centroids)
        # Cluster sizes
        for i in range(k):
            print(f"Cluster {i} size:", np.sum(labels == i))

        new_centroids = update_centroids(X, labels, k)
        print("Updated Centroids:\n", new_centroids)

        # Convergence check
        if np.allclose(centroids, new_centroids):
            print("Converged!")
            break

        centroids = new_centroids

    wcss = compute_wcss(X, labels, centroids)
    return centroids, labels, wcss

# Call function
# ↓
# Pick initial centroids
# ↓
# LOOP:
#     Assign clusters
#     Count cluster sizes
#     Update centroids
#     Check convergence
# ↓
# Compute WCSS
# ↓
# Return results
# -------- Run K-Means (Run 1) --------
k = 3
centroids1, labels1, wcss1 = kmeans(X_km, k, "first")

# Plot clusters
plt.figure()
plt.scatter(X_km[:,0], X_km[:,1], c=labels1, cmap='viridis')
plt.scatter(centroids1[:,0], centroids1[:,1], c='red', marker='X', s=200)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("K-Means Clustering (Run 1)")
plt.grid()
plt.show()

print("WCSS Run 1:", wcss1)


# -------- Map clusters to actual labels --------
mapping = {}
for i in range(k):
    labels_cluster = y[labels1 == i]
    if len(labels_cluster) > 0:
        mapping[i] = Counter(labels_cluster).most_common(1)[0][0]
# count = {}
# for val in labels_cluster:
#     if val in count:
#         count[val] += 1
#     else:
#         count[val] = 1
# max_label = None
# max_count = 0

# for key in count:
#     if count[key] > max_count:
#         max_count = count[key]
#         max_label = key
# mapping[i] = max_label
# -------------- if they ask us to implemnet wighted K-MEANS .. # ...
# mapping = {}

# for i in range(k):
#     indices = np.where(labels1 == i)[0]   # indices of points in cluster i
    
#     if len(indices) > 0:
#         weight_sum = {}   # label → total weight
        
#         for idx in indices:
#             label = y[idx]              # actual class
#             point = X_km[idx]           # data point
#             centroid = centroids1[i]    # centroid of cluster
            
#             # distance
#             dist = np.sum((point - centroid)**2)
            
#             # weight (IMPORTANT)
#             weight = 1 / (dist + 1e-6)   # avoid division by zero
            
#             if label in weight_sum:
#                 weight_sum[label] += weight
#             else:
#                 weight_sum[label] = weight
        
#         # find label with max weight
#         max_label = None
#         max_weight = 0
        
#         for key in weight_sum:
#             if weight_sum[key] > max_weight:
#                 max_weight = weight_sum[key]
#                 max_label = key
        
#         mapping[i] = max_label

        # it finds the most common trur-label in taht cluster-i ..


# pred_labels = np.array([mapping[c] for c in labels1])
pred_labels=[]
for c in labels1:
    pred_labels.append(mapping[c])

# accuracy = np.mean(pred_labels == y)
correct = 0
for i in range(len(y)):
    if pred_labels[i] == y[i]:
        correct += 1

accuracy = correct / len(y)
print("Clustering Accuracy:", accuracy)


# -------- Run K-Means (Run 2 - Random Init) --------
centroids2, labels2, wcss2 = kmeans(X_km, k, "random")

print("\nComparison:")
print("WCSS Run 1:", wcss1)
print("WCSS Run 2:", wcss2)


# ================================
# PART D: ELBOW METHOD
# ================================

wcss_values = []

for k_val in range(1, 7):
    _, _, wcss_val = kmeans(X_km, k_val, "first", max_iters=20)
    wcss_values.append(wcss_val)

# Plot Elbow Curve
plt.figure()
plt.plot(range(1,7), wcss_values, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.grid()
plt.show()

# compute_distances -- how far each point is from centroids
# assign_clusters -- which cluster each point belongs to
# update_centroids -- move centroids to center of clusters
# compute_wcss -- measure clustering quality
# kmeans -- runs entire process
# ---------------  ##############_------------VERSION--2   ---- ########################





# # -*- coding: utf-8 -*-
# """Assg_6.ipynb

# Automatically generated by Colab.

# Original file is located at
#     https://colab.research.google.com/drive/1DwWdUVRI1XRVvTRbjptNT3z0fVRGo39u
# """

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from sklearn.datasets import load_iris

# iris = load_iris()
# X = iris.data
# y = iris.target

# mean = np.mean(X, axis=0)
# X_centered = X - mean

# #PCA using covariance matrix

# n = X_centered.shape[0]
# c = (1/(n-1)) * np.dot(X_centered.T, X_centered)

# eigenvalues, eigenvectors = np.linalg.eig(c)

# idx = np.argsort(eigenvalues)[::-1]
# eigenvalues = eigenvalues[idx]
# eigenvectors = eigenvectors[:, idx]

# W = eigenvectors[:, :2]
# Z = np.dot(X_centered, W)

# plt.scatter(Z[:,0], Z[:,1], c=y, cmap="viridis", s=50)
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.title("Principal Component Analysis")
# plt.show()

# plt.scatter(X[:, 2], X[:, 3], c=y)
# plt.xlabel("petal-length")
# plt.ylabel("petal-width")
# plt.title("original-plot")
# plt.show()

# explained_variance_ratio = eigenvalues/(np.sum(eigenvalues))
# explained_variance_ratio

# for i,val in enumerate(explained_variance_ratio):
#     print(f"PC{i+1}: {val:.4f}")

# U, s, Vt = np.linalg.svd(X_centered)

# U.shape

# s.shape

# Vt.shape

# c1 = Vt.T[:, 0]
# c2 = Vt.T[:, 1]

# W2 = Vt.T[:, :2]
# Z2 = X_centered.dot(W2)

# plt.scatter(Z[:,0], Z[:,1], c=y, cmap="viridis", s=50)
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.title("Principal Component Analysis")
# plt.show()

# X3 = Z2.dot(W2.T) + mean

# plt.scatter(X[:,2], X[:,3], label="Original", alpha=0.5)
# plt.scatter(X3[:,2], X3[:,3], label="Reconstructed", alpha=0.5)
# plt.legend()
# plt.xlabel("petal-length")
# plt.ylabel("petal-width")
# plt.title("Reconstructed-plot")
# plt.show()

# error = np.linalg.norm(X - X3, 'fro')
# print("Reconstruction Error:", error)

# X_km = X[:, [2,3]]

# k = 3

# centroids = X_km[:k].copy()

# def compute_distance(a,b):
#     return np.linalg.norm(a - b)

# def assign_clusters(X, centroids):
#     clusters = []
#     for point in X:
#         distance = [compute_distance(point,c) for c in centroids]
#         clusters.append(np.argmin(distance))
#     return np.array(clusters)

# def update_centroids(X, clusters, k):
#     new_centroids = []
#     for i in range(k):
#         points = X[clusters == i]

#         if len(points) > 0:
#             new_centroids.append(np.mean(points, axis=0))
#         else:
#             new_centroids.append(X[np.random.randint(len(X))])
#     return np.array(new_centroids)

# for i in range(20):
#     clusters = assign_clusters(X_km, centroids)
#     new_centroids = update_centroids(X_km, clusters, k)
#     print(f"\nIteration {i+1}-----------------------------")
#     for j in range(k):
#         print(f"Cluster {j} size:", np.sum(clusters == j))
#         print(f"Centroid {j}:", new_centroids[j])

#     # Check convergence
#     if np.allclose(centroids, new_centroids):
#         print("Converged!")
#         break
#     centroids = new_centroids

# plt.scatter(X_km[:, 0], X_km[:, 1], c=clusters, cmap='viridis')
# plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200)
# plt.xlabel("Petal Length")
# plt.ylabel("Petal Width")
# plt.title("K-Means Clustering (k=3)")
# plt.show()

# #Within-Cluster Sum of Squares
# def compute_wcss(X, clusters, centroids):
#     wcss = 0
#     for i in range(len(X)):
#         c = centroids[clusters[i]]
#         wcss += np.sum((X[i] - c) ** 2)
#     return wcss
# wcss = compute_wcss(X_km, clusters, centroids)
# print(wcss)

# print("\nFirst 10 predicted clusters:", clusters[:10])
# print("First 10 actual labels:", y[:10])

# # Random initialization
# indices = np.random.choice(len(X_km), k, replace=False)
# centroids_rand = X_km[indices]

# for i in range(20):
#     clusters_rand = assign_clusters(X_km, centroids_rand)
#     new_centroids_rand = update_centroids(X_km, clusters_rand, k)

#     if np.allclose(centroids_rand, new_centroids_rand):
#         break

#     centroids_rand = new_centroids_rand

# # Compute WCSS
# wcss_rand = compute_wcss(X_km, clusters_rand, centroids_rand)

# print("\n--- Comparison ---")
# print("Initial (first 3 points) centroids:\n", centroids)
# print("Random centroids:\n", centroids_rand)
# print("WCSS (initial):", wcss)
# print("WCSS (random):", wcss_rand)

# from collections import Counter

# mapping = {}

# for i in range(k):
#     labels = y[clusters == i]
#     most_common = Counter(labels).most_common(1)[0][0]
#     mapping[i] = most_common
# print("cluster to label mapping: ", mapping)

# pred_labels = np.array([mapping[c] for c in clusters])
# pred_labels

# accuracy = np.sum(pred_labels == y)/len(y)
# print("Clustering Accuracy:", accuracy)

# print("\nSample comparison:")
# for i in range(10):
#     print(f"Cluster: {clusters[i]} → Predicted: {pred_labels[i]}, Actual: {y[i]}")

# from sklearn.metrics import confusion_matrix

# cm = confusion_matrix(y, pred_labels)
# print("\nConfusion Matrix:\n", cm)

# # Elbow _ Method

# wcss_values = []

# for k in range(1,7):
#     centroids = X_km[:k].copy()

#     for _ in range(20):
#         clusters = assign_clusters(X_km, centroids)
#         new_centroids = update_centroids(X_km, clusters, k)

#         if np.allclose(centroids, new_centroids):
#             break
#         centroids = new_centroids
#     wcss = compute_wcss(X_km, clusters, centroids)
#     wcss_values.append(wcss)
#     print(f"k = {k}, WCSS = {wcss}")

# plt.plot(range(1, 7), wcss_values, marker='o')
# plt.xlabel("Number of Clusters (k)")
# plt.ylabel("WCSS")
# plt.title("Elbow Method for Optimal k")
# plt.show()

############---------------------------- VERSION-3 ***************** ####################



# -*- coding: utf-8 -*-
"""Untitled11.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1WchKGwoKUKYYi0AcN3uZxPWQWErwXwxI
"""

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris

# # Load dataset
# iris = load_iris()
# X = iris.data   # shape (150, 4)
# y = iris.target
# feature_names = iris.feature_names

# print("Shape of X:", X.shape)

# # for our understanding clearly --- ** ##
# df = pd.DataFrame(X, columns=feature_names)
# df['target'] = y

# df.head()

# # Compute mean (1 x 4)
# mean_vector = np.mean(X, axis=0)
# print("Mean Vector:\n", mean_vector)

# # Center the dataset
# X_centered = X - mean_vector

# # Check (mean should be ~0)
# print("Mean after centering:\n", np.mean(X_centered, axis=0))
# # X_centered.shape

# n = X_centered.shape[0]

# cov_matrix = (1 / (n - 1)) * np.dot(X_centered.T, X_centered)

# print("Covariance Matrix:\n", cov_matrix)

# eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# print("Eigenvalues:\n", eigenvalues)
# print("Eigenvectors:\n", eigenvectors)

# # Sort indices
# sorted_idx = np.argsort(eigenvalues)[::-1]

# # Sort eigenvalues
# eigenvalues_sorted = eigenvalues[sorted_idx]

# # Sort eigenvectors
# eigenvectors_sorted = eigenvectors[:, sorted_idx]

# print("Sorted Eigenvalues:\n", eigenvalues_sorted)

# # Take first 2 eigenvectors
# W = eigenvectors_sorted[:, :2]

# print("Projection Matrix W:\n", W)

# X_pca = np.dot(X_centered, W)

# print("Shape after PCA:", X_pca.shape)

# plt.figure()

# for i in range(3):
#     plt.scatter(
#         X[y == i, 2],  # petal length
#         X[y == i, 3],  # petal width
#         label=f"Class {i}"
#     )

# plt.xlabel("Petal Length")
# plt.ylabel("Petal Width")
# plt.title("Original Data")
# plt.legend()
# plt.grid()

# plt.show()

# plt.figure()

# for i in range(3):
#     plt.scatter(
#         X_pca[y == i, 0],
#         X_pca[y == i, 1],
#         label=f"Class {i}"
#     )

# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.title("PCA Transformed Data")
# plt.legend()
# plt.grid()

# plt.show()

# explained_variance_ratio = eigenvalues_sorted / np.sum(eigenvalues_sorted)

# print("Explained Variance Ratio:\n", explained_variance_ratio)

# # For first 2 components
# print("Variance captured by 2 PCs:", np.sum(explained_variance_ratio[:2]))

# # Mean vector
# mean_vector = np.mean(X, axis=0)

# # Centering
# Xc = X - mean_vector

# # Check
# print("Mean after centering:", np.mean(Xc, axis=0))

# U, S, Vt = np.linalg.svd(Xc)

# print("U shape:", U.shape)
# print("Singular values:", S)
# print("V^T shape:", Vt.shape)

# print("\nMatrix U:\n", U)
# print("\nSingular Values (Σ):\n", S)
# print("\nMatrix V^T:\n", Vt)

# # Take first 2 principal directions
# V2 = Vt.T[:, :2]   # (4 x 2)

# # Transform data
# Z = np.dot(Xc, V2)

# print("Shape of Z:", Z.shape)

# plt.figure()

# for i in range(3):
#     plt.scatter(Z[y == i, 0], Z[y == i, 1], label=f"Class {i}")

# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.title("PCA using SVD (2D Projection)")
# plt.legend()
# plt.grid()

# plt.show()

# X_approx = np.dot(Z, V2.T)

# print("Shape of reconstructed X:", X_approx.shape)

# plt.figure()

# # Original
# plt.scatter(X[:, 2], X[:, 3], label="Original", alpha=0.6)

# # Reconstructed
# plt.scatter(X_approx[:, 2], X_approx[:, 3], label="Reconstructed", alpha=0.6)

# plt.xlabel("Petal Length")
# plt.ylabel("Petal Width")
# plt.title("Original vs Reconstructed Data")
# plt.legend()
# plt.grid()

# plt.show()

# error = np.linalg.norm(X - X_approx, 'fro')

# print("Reconstruction Error (Frobenius Norm):", error)

# # Covariance matrix
# cov_matrix = np.dot(Xc.T, Xc) / (Xc.shape[0] - 1)

# # Eigen decomposition
# eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

# # Sort
# idx = np.argsort(eig_vals)[::-1]
# eig_vecs = eig_vecs[:, idx]

# # Take top 2
# W = eig_vecs[:, :2]

# # Transform
# Z_cov = np.dot(Xc, W)

# # Compare visually
# plt.figure()

# plt.scatter(Z[:, 0], Z[:, 1], label="SVD PCA", alpha=0.6)
# plt.scatter(Z_cov[:, 0], Z_cov[:, 1], label="Covariance PCA", alpha=0.6)

# plt.legend()
# plt.title("SVD vs Covariance PCA")
# plt.grid()

# plt.show()

# X_km = X[:, [2, 3]]  # petal length, petal width

# k = 3
# centroids = X_km[:k].copy()

# print("Initial Centroids:\n", centroids)

# def compute_distance(X, centroids):
#     distances = np.zeros((X.shape[0], k))
#     for i in range(k):
#         distances[:, i] = np.sum((X - centroids[i])**2, axis=1)
#     return distances

# max_iters = 20

# for it in range(max_iters):
#     print(f"\nIteration {it+1}")

#     # Assignment Step
#     distances = compute_distance(X_km, centroids)
#     labels = np.argmin(distances, axis=1)

#     # Print cluster sizes
#     for i in range(k):
#         print(f"Cluster {i} size:", np.sum(labels == i))

#     # Update Step
#     new_centroids = np.array([
#         X_km[labels == i].mean(axis=0) if np.sum(labels == i) > 0 else centroids[i]
#         for i in range(k)
#     ])

#     print("Updated Centroids:\n", new_centroids)

#     # Convergence check
#     if np.allclose(centroids, new_centroids):
#         print("Converged!")
#         break

#     centroids = new_centroids

# plt.figure()

# for i in range(k):
#     plt.scatter(X_km[labels == i, 0], X_km[labels == i, 1], label=f"Cluster {i}")

# plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='X', s=200, label='Centroids')

# plt.xlabel("Petal Length")
# plt.ylabel("Petal Width")
# plt.title("K-Means Clustering")
# plt.legend()
# plt.grid()

# plt.show()

# wcss = 0
# for i in range(k):
#     wcss += np.sum((X_km[labels == i] - centroids[i])**2)

# print("WCSS:", wcss)

# print("\nFirst 20 Predictions vs Actual:")
# for i in range(20):
#     print(f"Predicted: {labels[i]}, Actual: {y[i]}")

# # Random initialization
# np.random.seed(42)
# rand_idx = np.random.choice(X_km.shape[0], k, replace=False)
# centroids2 = X_km[rand_idx]

# for it in range(max_iters):
#     distances = np.zeros((X_km.shape[0], k))
#     for i in range(k):
#         distances[:, i] = np.sum((X_km - centroids2[i])**2, axis=1)

#     labels2 = np.argmin(distances, axis=1)

#     new_centroids2 = np.array([
#         X_km[labels2 == i].mean(axis=0) if np.sum(labels2 == i) > 0 else centroids2[i]
#         for i in range(k)
#     ])

#     if np.allclose(centroids2, new_centroids2):
#         break

#     centroids2 = new_centroids2

# # Compute WCSS
# wcss2 = sum(np.sum((X_km[labels2 == i] - centroids2[i])**2) for i in range(k))

# print("New Centroids:\n", centroids2)
# print("New WCSS:", wcss2)

# print("\nComparison:")
# print("Run 1 WCSS:", wcss)
# print("Run 2 WCSS:", wcss2)

# print("\nCentroids Run 1:\n", centroids)
# print("\nCentroids Run 2:\n", centroids2)

# def kmeans_wcss(X, k):
#     centroids = X[:k].copy()

#     for _ in range(20):
#         distances = np.zeros((X.shape[0], k))
#         for i in range(k):
#             distances[:, i] = np.sum((X - centroids[i])**2, axis=1)

#         labels = np.argmin(distances, axis=1)

#         new_centroids = np.array([
#             X[labels == i].mean(axis=0) if np.sum(labels == i) > 0 else centroids[i]
#             for i in range(k)
#         ])

#         if np.allclose(centroids, new_centroids):
#             break

#         centroids = new_centroids

#     wcss = sum(np.sum((X[labels == i] - centroids[i])**2) for i in range(k))
#     return wcss

# k_values = range(1, 7)
# wcss_values = []

# for k_val in k_values:
#     wcss_values.append(kmeans_wcss(X_km, k_val))

# print("WCSS values:", wcss_values)

# plt.figure()

# plt.plot(k_values, wcss_values, marker='o')

# plt.xlabel("Number of Clusters (k)")
# plt.ylabel("WCSS")
# plt.title("Elbow Method")

# plt.grid()
# plt.show()
