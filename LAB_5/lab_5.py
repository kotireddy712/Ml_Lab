import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

X = np.array([[0,0], [0,1], [1,0], [1,1]])
AND_y = np.array([0,0,0,1])
OR_y = np.array([0,1,1,1])
XOR_y = np.array([0,1,1,0])

datasets = {"AND": AND_y, "OR": OR_y, "XOR": XOR_y}

def step(z):
    return np.where(z >= 0, 1, 0)

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

class SingleLayer:
    def __init__(self, activation):
        self.activation = activation
        self.w = np.random.randn(2) * 0.5
        self.b = np.random.randn() * 0.1

    def train(self, X, y, lr=0.1, epochs=5000):
        for _ in range(epochs):
            z = np.dot(X, self.w) + self.b
            
            if self.activation == step:
                y_pred = step(z)
                error = y - y_pred
                self.w += lr * np.dot(X.T, error)
                self.b += lr * np.sum(error)
            
            elif self.activation == sigmoid:
                a = sigmoid(z)
                error = a - y
                self.w -= lr * np.dot(X.T, error) / len(X)
                self.b -= lr * np.sum(error) / len(X)
            
            elif self.activation == relu:
                a = relu(z)
                error = (a - y) * relu_derivative(z)
                self.w -= lr * np.dot(X.T, error) / len(X)
                self.b -= lr * np.sum(error) / len(X)

    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        if self.activation == sigmoid:
            return (sigmoid(z) >= 0.5).astype(int)
        else:
            return step(z)

class TwoLayer:
    def __init__(self, activation, hidden_size=4, lr=0.1):
        self.activation = activation
        self.hidden_size = hidden_size
        self.lr = lr
            
        self.W1 = np.random.randn(2, hidden_size) * 0.5
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, 1) * 0.5 # 1 = no.of.classes.
        self.b2 = np.zeros(1) # 1 = no.of.classes..

    def train(self, X, y, epochs=15000):
        y = y.reshape(-1, 1)
        for _ in range(epochs):
            z1 = np.dot(X, self.W1) + self.b1
            a1 = self.activate(z1) # a1=hidden_actiavtion(z1)..
            z2 = np.dot(a1, self.W2) + self.b2
            a2 = self.activate(z2) # a2 = output_activation(z2)..
            
            if self.activation == step:
                delta2 = a2 - y
            else:
                delta2 = (a2 - y) * self.activate_derivative(z2) # output
            
            dW2 = np.dot(a1.T, delta2)
            db2 = np.sum(delta2, axis=0)
            
            if self.activation == step:
                delta1 = np.dot(delta2, self.W2.T)
            else:
                delta1 = np.dot(delta2, self.W2.T) * self.activate_derivative(z1) #hidden
            
            dW1 = np.dot(X.T, delta1)
            db1 = np.sum(delta1, axis=0)
            
            self.W2 -= self.lr * dW2 / len(X)
            self.b2 -= self.lr * db2 / len(X)
            self.W1 -= self.lr * dW1 / len(X)
            self.b1 -= self.lr * db1 / len(X)

    def activate(self, z):
        if self.activation == sigmoid:
            return sigmoid(z)
        elif self.activation == relu:
            return relu(z)
        else:
            return self.activation(z).astype(float)

    def activate_derivative(self, z):
        if self.activation == sigmoid:
            s = sigmoid(z)
            return s * (1 - s)
        elif self.activation == relu:
            return relu_derivative(z)
        else:
            return np.ones_like(z)

    def predict(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.activate(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.activate(z2)
        return (a2 >= 0.5).astype(int).flatten()

activations = {"Step": step, "Sigmoid": sigmoid, "ReLU": relu}

# %%
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:,0], X[:,1], c=y, s=100)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

for gate, y in datasets.items():
    print(f"\n========== {gate} ==========")
    for name, act in activations.items():
        # Single Layer
        sl = SingleLayer(act)
        sl.train(X, y)
        sl_acc = np.mean(sl.predict(X) == y)
        print(f"Single Layer - {name}: {sl_acc:.2%}")
        plot_decision_boundary(sl, X, y, f"{gate} - Single Layer - {name}")
        
        # Two Layer with optimized learning rate for sigmoid
        lr = 0.5 if act == sigmoid else 0.1
        tl = TwoLayer(act, hidden_size=4, lr=lr)
        tl.train(X, y, epochs=20000 if act == sigmoid else 15000)
        tl_acc = np.mean(tl.predict(X) == y)
        print(f"Two Layer - {name}:  {tl_acc:.2%}")
        plot_decision_boundary(tl, X, y, f"{gate} - Two Layer - {name}")



# %%




# %%