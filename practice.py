import numpy as np
import pandas as pd
import matplotlib as plt

# ----------------------------
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from skelarn.model_selection import train_test_split


#------------------------------
cols = ["c1","c2","c3","c4","class"]
df = pd.read_csv("path_of_file",names = cols,header=None)
df.head() # to print-first 5- rows in dataset of csv file..
X = df.drop("class",axis=1)
Y = df["class"]

# Label Encoding ..
le = LabelEncoder()
X_LabelEncoded = le.fit_transform(X) # label-encoding done ..
Y_LabelEncoded = le.fit_transform(Y)

# one-hot-encoding ..

OHE = OneHotEncoder(sparse_output = False)
X_onehot = OHE.fit_transform(X)
le1 = LabelEncoder()
y_LE = le1.fit_transform(le1)
Y_onehot= np.eye(len(np.unique(y_LE)))[y_LE]

# discretization / Binning ..
for col in X.columns:
    X[col] = pd.cut(X[col],bins=3,labels=[0,1,2])

X = X.astype(int)

# if some-rows have to removed then ::
df = df[df["class"] != 2] # !=, ==, >=,<=,..

# Train - Test - Split ..
X_train,X_test,Y_train,Y_test = train_test_split(X_onehot,Y_onehot,test_size = 0.2,random_state = 42,stratify = y_LE)
 # here itself we can divide test data in to = test+ validation by using startify as conveting back to one-hot to label encoding by - np.argmax(y,axis=1)


# ---------------------------------------------------------------- ## 
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,log_loss

cols=["c1","c2","output"]
df = pd.read_csv("file_name",names=cols)

X = df.drop("output",axis=1)
Y = df["output"]

# features (categorical → one-hot)
OHE = OneHotEncoder(sparse_output = False)
X_enc = OHE.fit_transform(X)

# features (categorical → one-hot)
le = LabelEncoder()
y_enc = le.fit_transform(Y)

# first split test
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_enc, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)

# split validation from training
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25,
    stratify=y_trainval, random_state=42
)

# Find Optimal Number of Hidden Layers
        #Keep neurons fixed (e.g., 16 each).
neurons = 16
layer_options = [1,2,3,4]
train_loss = []
val_loss = []
for layer in layer_options:
    arch = tuple([neurons]*layer) # hidden_layer_sizes only accept tuples not-list == ()
    model = MLPClassifier (
        hidden_layer_sizes = arch,
        activation = "relu", #"sigmod" # "tanh" ,...
        solver = "adam", # defalut ..
        max_iter = 1000,
        learning_rate_init = 0.0001, # defalut
        random_state = 42
    )
    model.fit(X_train,y_train)
    #training-porb..
    train_porb = model.predict_proba(X_train)
    train_loss.append(log_loss(y_train,train_porb))
    # similar for val.loss
    val_prob = model.predict_proba(X_val)
    val_loss.append(log_loss(y_val,val_prob))

# plot layer-search 
layers = [1,2,3,4]
# Validation loss determines optimal layer count.
plt.plot(layers,val_loss,marker = 'o',title = "validation-loss")
plt.xlabel("no.of.hidden layers")
plt.ylabel("VAL_LOSS")
plt.legend()
plt.show()
best_idx = np.argmin(val_loss)
best_layers = layer_options[best_idx]
print(best_layers)

# Find Optimal Nodes
    #Fix the number of layers and vary neurons.
nodes = [4,8,16,32,64]
train_loss = []
val_loss = []
for node in nodes:
    arch = tuple([node]*best_layers)
    model = MLPClassifier(
        hidden_layer_sizes = arch,
        activation = "relu",
        solver = "adam",
        max_iter = 1000,
        learning_rate_init = 0.0001,
        random_state = 42
    )
    model.fit(X_train,y_train)
    #training-porb..
    train_porb = model.predict_proba(X_train)
    train_loss.append(log_loss(y_train,train_porb))
    # similar for val.loss
    val_prob = model.predict_proba(X_val)
    val_loss.append(log_loss(y_val,val_prob))
  
plt.plot(nodes, train_loss, marker='o', label="Train Loss")
plt.plot(nodes, val_loss, marker='o', label="Validation Loss")

plt.xlabel("Nodes per Hidden Layer")
plt.ylabel("Loss")
plt.title("Node Selection")
plt.legend()
plt.show()

best_idx = np.argmin(val_loss)
best_nodes = nodes[best_idx]

final_arch = tuple([best_nodes]*best_layers)

final_model = MLPClassifier(
    hidden_layer_sizes = final_arch,
    activation = "relu",
    solver= "adam",
    max_iter = 1000,
    learning_rate_init = 0.001,
    random_state = 42
)

final_model.fit(X_trainval,y_trainval)

y_pred = final_model.predict(X_test)

print(classification_report(y_test,y_pred))

# -------------------------- S.V.M  ---------------------------------

from sklearn.svm import SVC

model = SVC(kernel = "rbf") # "linear","ploy"-degree = "",..
model.fit(X_trainval,y_train)
y_pred = model.predict(X_test)
model.support_vectors_

inidc = model.support_ # indices of support vector points in the dataset ..
X_sv = X[inidc]
Y_sv = Y[inidc]
# if we train on only X_Sv and Y_sv we get more accuaracy no-need on whole train dataset
# since, these points only mostly contibute to classificatioon ..
model.n_support_  ##== NO.OF.SUPPORT vectors in each class ..

# how to draw margins and descion boundaries and formual for max.width bettwen amgrins also ??

# ----------------------- D.T AND R.F # ------------------------------

pairs = [(0,1),(0,2)]

for i in range(len(pairs)):
    pair = pairs[i]
    plot_descion_boundary(X,y,model,pair,feature_names)
    plt.title(feature_names[pair[0]]+"vs"+feature_names[pair[1]])

#linspace
#meshgrid
#np.c_ reval

# criterion
# max_depth
# max_features
# n_estimators




