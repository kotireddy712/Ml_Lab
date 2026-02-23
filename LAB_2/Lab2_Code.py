import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.naive_bayes import CategoricalNB
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"

cols = ["buying","maint","doors","persons","lug_boot","safety","class"]
df = pd.read_csv(url, header=None, names=cols)

df.head()

le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

df.head()

X = df.drop("class", axis=1)
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def Prior(y):
    prior ={}
    length = len(y)
    counts = Counter(y)
    for c in counts :
        prior[c] = counts[c]/len(y)
    return prior

def train_nb(X, y):
    model = {}
    model["priors"] = Prior(y)
    model["likelihoods"] = {}

    for c in y.unique():
        model["likelihoods"][c] = {}
        Xc = X[y == c] # class-c rows stored in Xc : y==c boolean mask..
        for col in X.columns:
            values = list(Xc[col])
            model["likelihoods"][c][col] = Counter(values) # count freq only, not probabiltes ...
    return model
# -------------------------------
model = train_nb(X_train,y_train)
for c in model["likelihoods"]:
    for col in model["likelihoods"][c]:
        total_val = sum(model["likelihoods"][c][col].values())
        for v in model["likelihoods"][c][col]:
            model["likelihoods"][c][col][v] /= total_val

model["likelihoods"]
# ---------------------------------
# def predict_nb(model, X):
#     preds = []
#     for _, row in X.iterrows():
#         probs = {}
#         for c in model["priors"]:
#             prob = model["priors"][c]
#             for col in X.columns:
#                 prob *= model["likelihoods"][c][col].get(row[col], 1e-6) # if row[col] == prob - 0, gets deafult value..
#             probs[c] = prob
#         preds.append(max(probs, key=probs.get))
#     return np.array(preds)

# def train_gaussian_nb(X, y):
#     model = {}
#     model["priors"] = Prior(y)
#     model["mean"] = {}
#     model["var"] = {}

#     for c in set(y):
#         Xc = X[y == c]
#         model["mean"][c] = Xc.mean().to_dict()
#         model["var"][c]  = Xc.var().to_dict()

#     return model

# def predict_gaussian_nb(model, X):
#     predictions = []

#     for _, row in X.iterrows():
#         best_class = None
#         best_score = -1

#         for c in model["priors"]:
#             score = model["priors"][c]

#             for col in X.columns:
#                 mean = model["mean"][c][col]
#                 var  = model["var"][c][col]
#                 score *= gaussian_prob(row[col], mean, var)

#             if score > best_score:
#                 best_score = score
#                 best_class = c

#         predictions.append(best_class)

#     return np.array(predictions)

# import math

# def gaussian_prob(x, mean, var):
#     exponent = math.exp( -( (x - mean) ** 2 ) / (2 * var) )
#     coefficient = 1 / math.sqrt(2 * math.pi * var)
#     return coefficient * exponent

def predict_nb(model, X):
    predictions = []
    for _, row in X.iterrows():          # one test sample
        best_class = None
        best_score = -1

        for c in model["priors"]:        # try each class

            score = model["priors"][c]   # start with P(c)

            for col in X.columns:        # multiply likelihoods
                value = row[col]
                score *= model["likelihoods"][c][col].get(value, 1e-6)

            if score > best_score:        # keep max
                best_score = score
                best_class = c

        predictions.append(best_class)

    return np.array(predictions)


nb_model = train_nb(X_train, y_train)

y_train_pred = predict_nb(nb_model, X_train)
y_test_pred = predict_nb(nb_model, X_test)

pd.DataFrame({
    "True Label": y_test.values,
    "Predicted Label": y_test_pred
})

cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)
#cm_train
cm_test

def metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='macro'),
        "Recall": recall_score(y_true, y_pred, average='macro'),
        "F1": f1_score(y_true, y_pred, average='macro')
    }

print("Train Metrics :", metrics(y_train, y_train_pred))
print("Test Metrics :", metrics(y_test, y_test_pred))

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

nb_model_sm = train_nb(X_train_sm, y_train_sm)

y_train_sm_pred = predict_nb(nb_model_sm, X_train_sm)
y_test_sm_pred  = predict_nb(nb_model_sm, X_test)

cm_test_sm  = confusion_matrix(y_test, y_test_sm_pred)
print("Test Confusion Matrix (SMOTE):\n", cm_test_sm)

print("Train Metrics (SMOTE):", metrics(y_train_sm, y_train_sm_pred))
print("Test Metrics (SMOTE):", metrics(y_test, y_test_sm_pred))

# --------- Built-in Naive Bayes (Before SMOTE) ---------

nb_builtin = CategoricalNB()
nb_builtin.fit(X_train, y_train)

y_train_pred = nb_builtin.predict(X_train)
y_test_pred  = nb_builtin.predict(X_test)

print("Built-in NB Confusion Matrix (Train)")
print(confusion_matrix(y_train, y_train_pred))

print("\nBuilt-in NB Confusion Matrix (Test)")
print(confusion_matrix(y_test, y_test_pred))

print("\nBuilt-in NB Metrics (Train):", metrics(y_train, y_train_pred))
print("Built-in NB Metrics (Test) :", metrics(y_test, y_test_pred))

# --------- Built-in Naive Bayes (After SMOTE) ---------

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

nb_builtin_sm = CategoricalNB()
nb_builtin_sm.fit(X_train_sm, y_train_sm)

y_train_sm_pred = nb_builtin_sm.predict(X_train_sm)
y_test_sm_pred  = nb_builtin_sm.predict(X_test)

print("\nBuilt-in NB AFTER SMOTE Confusion Matrix (Train)")
print(confusion_matrix(y_train_sm, y_train_sm_pred))

print("\nBuilt-in NB AFTER SMOTE Confusion Matrix (Test)")
print(confusion_matrix(y_test, y_test_sm_pred))

print("\nBuilt-in NB AFTER SMOTE Metrics (Train):", metrics(y_train_sm, y_train_sm_pred))
print("Built-in NB AFTER SMOTE Metrics (Test) :", metrics(y_test, y_test_sm_pred))

# 2 - for the raw-implemnetation of KNN ..
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def get_neighbors(X_train, y_train, test_point, k):
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(X_train[i], test_point)
        distances.append((dist, y_train[i]))

    distances.sort(key=lambda x: x[0]) #key = lambda x:x[0]
    neighbors = distances[:k]
    return neighbors

# def predict_classification(neighbors):
#     labels = [label for _, label in neighbors]
#     most_common = Counter(labels).most_common(1)
#     return most_common[0][0]

def predict_classification(neighbors):
    counts = {}
    # count each label
    for i in range(len(neighbors)):
        label = neighbors[i][1]

        if label in counts:
            counts[label] = counts[label] + 1
        else:
            counts[label] = 1
    # find label with maximum count
    best_label = None
    max_count = -1
    for label in counts:
        if counts[label] > max_count:
            max_count = counts[label]
            best_label = label
    return best_label


def knn_predict(X_train, y_train, X_test, k):
    predictions = []

    for test_point in X_test:
        neighbors = get_neighbors(X_train, y_train, test_point, k)
        pred_label = predict_classification(neighbors)
        predictions.append(pred_label)

    return np.array(predictions)

y_pred_k1 = knn_predict(
    X_train.values,
    y_train.values,
    X_test.values,
    k=1
)

k_sqrt = int(np.sqrt(len(X_train)))

y_pred_ksqrt = knn_predict(
    X_train.values,
    y_train.values,
    X_test.values,
    k=k_sqrt
)

y_pred_k60 = knn_predict(
    X_train.values,
    y_train.values,
    X_test.values,
    k=60
)

print("k = 1     :", metrics(y_test, y_pred_k1))
print("k = sqrt  :", metrics(y_test, y_pred_ksqrt))
print("k = 60    :", metrics(y_test, y_pred_k60))

#2 - IN BUILT FOR KNN
def evaluate(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro"),
        "Recall": recall_score(y_true, y_pred, average="macro"),
        "F1": f1_score(y_true, y_pred, average="macro")
    }
knn_1 = KNeighborsClassifier(n_neighbors=1)
knn_1.fit(X_train, y_train)

y_pred_1 = knn_1.predict(X_test)
metrics_k1 = evaluate(y_test, y_pred_1)

metrics_k1

k_sqrt = int(np.sqrt(len(X_train)))

knn_sqrt = KNeighborsClassifier(n_neighbors=k_sqrt)
knn_sqrt.fit(X_train, y_train)

y_pred_sqrt = knn_sqrt.predict(X_test)
metrics_ksqrt = evaluate(y_test, y_pred_sqrt)

metrics_ksqrt

knn_60 = KNeighborsClassifier(n_neighbors=60)
knn_60.fit(X_train, y_train)

y_pred_60 = knn_60.predict(X_test)
metrics_k60 = evaluate(y_test, y_pred_60)

metrics_k60

pd.DataFrame({
    "k = 1": metrics_k1,
    "k = sqrt(n)": metrics_ksqrt,
    "k = 60": metrics_k60
})

error_rate = []

for k in range(1, 41):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    pred_k = knn.predict(X_test)
    error_rate.append(1 - accuracy_score(y_test, pred_k))

# plt.figure(figsize=(8,5))
plt.plot(range(1,41), error_rate, marker='o')
plt.xlabel("K value")
plt.ylabel("Error rate")
plt.title("Elbow Method for Optimal K")
plt.show()