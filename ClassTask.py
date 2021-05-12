from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import pandas as pd
from tqdm.notebook import tqdm
from scipy.special import softmax
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions

iris = datasets.load_iris()
X = iris.data[:,:2]
class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            loss = self.__loss(h, y)
                
            if(self.verbose ==True and i % 10000 == 0):
                print(f'loss: {loss} \t')
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X):
        return self.predict_prob(X).round()


iris.target_names
model_setosa = LogisticRegression(lr=0.1,num_iter=300000)
model_versicolor = LogisticRegression(lr=0.1,num_iter=300000)
model_virginica = LogisticRegression(lr=0.1,num_iter=300000)

y1 = (iris.target == 0)*1
model_setosa.fit(X,y1)
y2 = (iris.target == 1)*1
model_versicolor.fit(X,y2)
y3 = (iris.target == 2)*1
model_virginica.fit(X,y3)

plt.figure(figsize=(10,6))
plt.scatter(X[y1 == 0][:,0],X[y1 == 0][:,1],color='b',label='0')
plt.scatter(X[y1 == 1][:,0],X[y1 == 1][:,1],color='r',label='1')
plt.legend()
x1_min, x1_max = X[:,0].min(), X[:,0].max()
x2_min, x2_max = X[:,1].min(), X[:,1].max()
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
grid = np.c_[xx1.ravel(), xx2.ravel()]
probs = model_setosa.predict_prob(grid).reshape(xx1.shape)
plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors='black');

plt.figure(figsize=(10,6))
plt.scatter(X[y2 == 0][:,0],X[y2 == 0][:,1],color='b',label='0')
plt.scatter(X[y2 == 1][:,0],X[y2 == 1][:,1],color='r',label='1')
plt.legend()
x1_min, x1_max = X[:,0].min(), X[:,0].max()
x2_min, x2_max = X[:,1].min(), X[:,1].max()
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
grid = np.c_[xx1.ravel(), xx2.ravel()]
probs = model_versicolor.predict_prob(grid).reshape(xx1.shape)
plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors='black');

plt.figure(figsize=(10,6))
plt.scatter(X[y3 == 0][:,0],X[y3 == 0][:,1],color='b',label='0')
plt.scatter(X[y3 == 1][:,0],X[y3 == 1][:,1],color='r',label='1')
plt.legend()
x1_min, x1_max = X[:,0].min(), X[:,0].max()
x2_min, x2_max = X[:,1].min(), X[:,1].max()
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
grid = np.c_[xx1.ravel(), xx2.ravel()]
probs = model_virginica.predict_prob(grid).reshape(xx1.shape)
plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors='black');

plt.figure(figsize=(10,6))
plt.scatter(X[iris.target == 0][:,0],X[iris.target == 0][:,1],color='b',label='0')
plt.scatter(X[iris.target == 1][:,0],X[iris.target == 1][:,1],color='r',label='1')
plt.scatter(X[iris.target == 2][:,0],X[iris.target == 2][:,1],color='y',label='2')
plt.legend()
x1_min, x1_max = X[:,0].min(), X[:,0].max()
x2_min, x2_max = X[:,1].min(), X[:,1].max()
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
grid = np.c_[xx1.ravel(), xx2.ravel()]
probs1 = model_setosa.predict_prob(grid).reshape(xx1.shape)
probs2 = model_versicolor.predict_prob(grid).reshape(xx1.shape)
probs3 = model_virginica.predict_prob(grid).reshape(xx1.shape)
plt.contour(xx1, xx2, probs1, [0.5], linewidths=1, colors='blue');
plt.contour(xx1, xx2, probs2, [0.5], linewidths=1, colors='red');
plt.contour(xx1, xx2, probs3, [0.5], linewidths=1, colors='yellow');

iris = datasets.load_iris()
X_train = iris.data[:,:2]
Y_train = iris.target

_,X_test,_,y_test = train_test_split(X_train,Y_train)

n_neighbors = 25
model = KNeighborsClassifier(n_neighbors=n_neighbors)
model.fit(X_train,Y_train)

fig, gs = plt.figure(figsize=(9,4)), gridspec.GridSpec(1, 2)
ax = []
for i in range(2):
    ax.append(fig.add_subplot(gs[i]))


plot_decision_regions(X_train, Y_train, model, ax=ax[0])
plot_decision_regions(X_test, y_test, model, ax=ax[1])
plt.show()
y_pred = model.predict(X_test)
classification_report(y_test,y_pred)

