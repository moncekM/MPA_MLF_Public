import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

iris=load_iris()
iris.feature_names
print(iris.feature_names)
print(iris.data[0:5,:])
print(iris.target[0:5])
#print(iris.data)

X=iris.data[iris.target!=2,0:2]
y=iris.target[iris.target!=2]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
print(X_train.shape)
print(X_test.shape)

SVMmodel=SVC(kernel='linear')
SVMmodel.fit(X_train,y_train)
SVMmodel.get_params()
SVMmodel.score(X_test,y_test)

print(iris.data[:,0:2])
print(iris.target == 2)

supvectors=SVMmodel.support_vectors_
# Plot the support vectors here

#Separating line coefficients:
W=SVMmodel.coef_
b=SVMmodel.intercept_
print(W)
print(b)
x1=np.linspace(np.min(X[:,0]),np.max(X[:,0]),100)
x2=-b/W[0,1]-W[0,0]/W[0,1]*x1
#plt،scatter(xf:,0j.l:,1)
plt. scatter (X[y==0,0],X[y==0,1],color="blue")
plt. scatter (X[y==1,0],X[y==1,1],color="red")
plt.scatter(x1,x2,color='black')
plt.show()













