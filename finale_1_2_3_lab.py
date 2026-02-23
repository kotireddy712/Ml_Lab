import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,precision_score,recall_score
from imblearn.over_sampling import SMOTE
import math

url = "path in question"

#if column names not given we add manually ..
cols = ["c1","c2","c3"]
df = pd.read_csv(url,header=None,names = cols)
# else - names = cols is an wastage..

# convert all of them to labels ( if we want to deal in labels..)...
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

X = df.iloc[:,:-1] # use .values if we want to chnage to numpy 

# else if we want to make some rows removed ..
X = df.drop("class-name",axis = 1)

# -- if we want to make polymnail FOR over basing there..
X = np.c_[X,X**2] 

Y = df.iloc[:,-1] # use .reshape(-1,1) later to use .. (m*1)- shape..

# np.random.seed(42)
#indicies = np.random.permuatation(len(X))
# X = X[indices]
# X_train = X[:100]
# ...
scaler = StandardScaler()
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 42,stratify = Y)

# if we want to handle in numerical then chnage in to all in same range..

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test) # but here fitted done by of tarining data only..

m,n = X.shape # m - no.of data samples, n = no.of.features..

# adding bais ...
X_train = np.c_[np.ones((X_train.shape[0],1)),X_train]
X_test = np.c_[np.ones((X_test.shape[0],1)),X_test]

# metrics if asked...
def metrics(y_true,y_pred):
    return {
        "Accuracy": accuracy_score(y_true,y_pred),
        "Precison": precision_score(y_true,y_pred,average = "macro") # sometimes..
        # "f1_score": f1_score(..), simailary all...

    }

#synthetic minimality oversampling techinque ( SMOTE )..
# from imblearn_oversampling import SMOTE..

smote = SMOTE(random_state = 42)
X_train,Y_train = smote.fit_resample(X_train,Y_train)

# 1.  linear-regression loss..

def compute_loss(X,Y,teta):
    m,n = X.shape
    y_pred = np.dot(X,teta)
    return ((1)/(2*m)) * np.mean((y_pred- Y) ** 2)

# 2. sigmoid regression losss..
def compute_loss_1(Y_pred,Y):
    eps = 1e-9
    return -np.mean( ( Y * np.log(Y_pred+eps) ) + ( 1-Y ) * ( np.log(1-Y_pred+eps) ) )

# compute sigmoid fucntion..
def sigmoid(Z):
    Z = np.clip(Z,-500,500)
    return 1/( 1+ np.exp(-Z) )
 
# compute label class in KNN..
def Predict_Label(X,teta,threshold = 0.5):
    Z = np.dot(X,teta)
    probs = sigmoid(Z)
    return (probs >= threshold).astype(int) # true label - 1 : above threshold, # false label = 0

# eculidea distance b.w two points taken as deafult..
def eculidean_distance(X1,X2):
    return np.sqrt( np.sum( (X1-X2) ** 2 ) )

# find the smallest k- neighbours distances..

def find_neighbours(X,Y,test_point,k):
    neighbours = []
    for i in range(len(X)):
        dist = eculidean_distance(X[i],test_point)
        neighbours.append((list,Y[i]))
    neighbours.sort(key = lambda x : x[0]) # deafult sort aslo no issue .. since dist is first in pair..
    points  = neighbours[:k]
    return points

# find the max.label among the k-nearest neighbours which ahs most weighted .....

def max_label(distances):
    temp = {}
    for dist,label in distances:
        if label in temp:
            temp[label] = temp[label] + 1
        else:
            temp[label] = 1
    best_label = None
    best_count = -1
    for label,counts in temp:
        if counts > best_count :
            best_label = label
            best_count = counts
    return best_label

# graident-descent for linear-regerssion.. X_train here already bais----added
def gradient_descent(X_train,Y_train,teta,lr = 0.001,iterations = 1000):
    m,n = X.shape
    teta = np.zeros((n,1))
    cost_list = []
    for i in range(iterations):
        Y_pred = np.dot(X_train,teta)
        dw = (1/m) * np.dot(X_train.T,Y_pred-Y_train)
        cost_list.append(compute_loss(X_train,Y_train,teta))
        teta = teta - ( lr * dw)
    return teta,cost_list

# gardient-descent for logistic regression ..
def graident_descent1(X_train,Y_train,teta,lr = 0.001,iterations = 1000):
    m,n = X.shape
    cost_list = []
    teta = np.zeros((n,1))
    for i in range(iterations):
        Z =np.dot(X_train,teta)
        Y_pred = sigmoid(Z)
        dw = (1/m)*np.dot(Y_pred,Y_train-Y_pred)
        teta = teta - (lr*dw)
        cost_list.append(compute_loss_1(Y_pred,Y_train))
    return teta,cost_list

# now we are entering in to gaint coding .. NAVIE - BAYERS..
def Prior(Y_train):
    priors = {}
    l = len(Y_train)
    cntr = Counter(list(Y_train))
    for c in cntr:
        priors[c] = cntr[c]/l
    return priors
# model of conating orior-porbbailtesm, we have still only counts of likelihoods not probbailtes..
def navie_bayes_model(X_train,Y_train):
    model = {}
    model["prior"] = Prior(Y_train)
    model["likelihood"] = {}
    for c in set(Y_train):
        model["likelihood"][c] = {}
        Xc = X[Y_train == c]
        for col in X.columns:
            model["likelihood"][c][col] = Counter(list(Xc[col]))
    return model

# converting their counts opf posteruior to probbailties.. so easy at end ..
#  1. first find the total-frequency of each class of [col]- each..
# 2, then divide it my each varite in that class after class 
def prob_likelihood(model):
    for c in model["likelihoods"]:
        for col in model["likelihoods"][c]:
           total_sum = sum(model["likelihood"][c][col].values())
           for varaite in model["likelihood"][c][col]:
                model["likelihood"][c][col][varaite] /= total_sum

# predicting the model of X-test..
def predict_model(model,X_test):
    finale = []
    for _,row in X_test.iterrows():
        best_class = None
        best_prob = -1
        for c in model["prior"]:
            probs = model["prior"][c]
            for col in X_train.columns:
                value = row[col] # acces the x_test value of each column ..
                probs = probs * model["likelihood"][c][col].get(value,1e-9)
            if probs > best_prob :
                best_class = c
                best_prob = probs
        finale.append((best_class,best_prob))
    return np.array(finale)
# main info. .iterrows() and then acces values of each in clomuns wise ...

# if they given values instaed of labels and asked used to follow guassain distru..

# same - priors ..
# bUT HERE :: instaed of likehoods ..
#  we have :: model["mean"]
#             model["var"] -  for each class and column-pairs..

# guassain - dist. formula

def guassain(x,mean,var):
    coff =  (1 / math.sqrt(2 * math.pi * var) )
    expo = math.exp( - ( (x-mean) ** 2 ) / (2 * var) )
    return coff * expo

# xc.mean().to_dict().. == model["mean"][c][col].. row[col] this time sent f0r gaus.porb..
# similary for xc.var()



# X = np.linspace(min,max,no.of_values)
# plt.plot(y_label)
# plt.plot(range(1,21),losses)
# plt.xlabel("---")
# plt.y_label("---")
# plt.title("---")
# plt.show() ##
#plt.scatter(x,y)

x_min,x_max =   X[:,0].min(),X[:,0].max()
y_min,y_max = ..
xx,yy = np.,meshgrid(np.linspace(),np.linspace)
np.c_[xx.reval(),yy.reval()]
