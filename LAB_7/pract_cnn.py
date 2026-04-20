from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,FLatten,Dense
import numpy as np

images =[]
labels = []
n_each_label = 50
for i in range(0,10):
    for j in range(n_each_label):
        temp = np.zeros((1,10))
        # if digit > 0 ::
        indic = np.random.choice(10,i,replace=False)
        temp[0,indic]=1
        images.append(temp)
        lablela.append(i)
    
return np.array(images),np.array(labels)

# reshape for CNN (VERY IMPORTANT)
X = X.reshape(-1, 1, 10, 1)

model = Sequential ([
        Conv2D(filters=3,kernel_size = (1,3),activation = 'relu',input_shape = (1,10)),
        # another conv.layer 
        MaxPooling2D(pool_size = (1,4)),
        Flatten(),
        Dense(32,activation='relu'),
        Dense(10,activation='softmax')
])

model.compile(optimizer='adam',loss='sparse_categorical_ceossentropy',metrics=['accuracy'])

model.fit(X,Y,epochs=100)

loss,acc = model.evaluate(xtest,ytest)
sample = x_test[0].reshae(1,1,10,1)
class_label = np.argmax(model.predict(sample))