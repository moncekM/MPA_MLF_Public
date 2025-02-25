import numpy as np
from scipy.constants import R
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn import preprocessing, __all__, decomposition
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Load Iris dataset as in the last PC lab:
iris=load_iris()
iris.feature_names
iris.feature_names
print(iris.feature_names)
print(iris.data[0:5,:])
print(iris.target[:])
# We have 4 dimensions of data, plot the first three colums in 3D
X=iris.data
y=iris.target
axes1=plt.axes(projection='3d')
axes1.scatter3D(X[y==0,1],X[y==0,1],X[y==0,2],color='green')
axes1.scatter3D(X[y==1,1],X[y==1,1],X[y==1,2],color='blue')
axes1.scatter3D(X[y==2,1],X[y==2,1],X[y==2,2],color='magenta')
plt.show()
# Pre-processing is an important step, you can try either StandardScaler (zero mean, unit variance of features)
# or MinMaxScaler (to interval from 0 to 1)
Xscaler = StandardScaler()
#Xscaler = MinMaxScaler()
Xpp=Xscaler.fit_transform(X)

# define PCA object (three components), fit and transform the data
pca = decomposition.PCA(n_components=3)
pca.fit(Xpp)
Xpca = pca.transform(Xpp)
print(pca.get_covariance())
# you can plot the transformed feature space in 3D:
axes2=plt.axes(projection='3d')
axes2.scatter3D(Xpca[y==0,0],Xpca[y==0,1],Xpca[y==0,2],color='green')
axes2.scatter3D(Xpca[y==1,0],Xpca[y==1,1],Xpca[y==1,2],color='blue')
axes2.scatter3D(Xpca[y==2,0],Xpca[y==2,1],Xpca[y==2,2],color='magenta')
plt.show()

# Compute pca.explained_variance_ and pca.explained_cariance_ratio_values
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)

# Plot the principal components in 2D, mark different targets in color
plt.scatter(Xpca[y==0,0],Xpca[y==0,1],color='green')
plt.scatter(Xpca[y==1,0],Xpca[y==1,1],color='blue')
plt.scatter(Xpca[y==2,0],Xpca[y==2,1],color='magenta')
plt.show()

# Import train_test_split as in last PC lab, split X (original) into train and test, train KNN classifier on full 4-dimensional X
X_train, X_test, y_trian, y_test = train_test_split(Xpp, y, test_size=0.3)
print(X_train.shape)
print(X_test.shape)
knn1=KNeighborsClassifier(n_neighbors = 3)
knn1.fit(X_train,y_trian)
Ypred=knn1.predict(X_test)
# Import and show confusion matrix
print(confusion_matrix(y_test,Ypred))
ConfusionMatrixDisplay.from_predictions(y_test,Ypred)

# Import train_test_split, split X after pca into train and test,
# train KNN classifier train KNN classifier on full 4-dimensional X pca network
X_trainpca, X_testpca, y_trianpca, y_testpca = train_test_split(Xpca, y, test_size=0.3)
print(X_trainpca.shape)
print(X_testpca.shape)
knn2=KNeighborsClassifier(n_neighbors = 3)
knn2.fit(X_trainpca,y_trianpca)
Ypredpca=knn2.predict(X_testpca)
# Import and show confusion matrix
print(confusion_matrix(y_testpca,Ypredpca))
ConfusionMatrixDisplay.from_predictions(y_testpca,Ypredpca)

# Import train_test_split, split X after pca into train and test,
# train KNN classifier on reduce 2 dimensions network with the split in most meaningfull way.
X_trainpca2, X_testpca2, y_trianpca2, y_testpca2 = train_test_split(Xpca[:,0:2], y, test_size=0.3)
print(X_trainpca2.shape)
print(X_testpca2.shape)
knn2=KNeighborsClassifier(n_neighbors = 3)
knn2.fit(X_trainpca2,y_trianpca2)
Ypredpca2=knn2.predict(X_testpca2)
# Import and show confusion matrix
print(confusion_matrix(y_testpca2,Ypredpca2))
ConfusionMatrixDisplay.from_predictions(y_testpca2,Ypredpca2)

# Import train_test_split, split X after pca into train and test,
# train KNN classifier on reduce 2 dimensions network with the split in not so meaningful way to sea the diference
X_trainpca3, X_testpca3, y_trianpca3, y_testpca3 = train_test_split(Xpca[:,1:3], y, test_size=0.3)
print(X_trainpca3.shape)
print(X_testpca3.shape)
knn3=KNeighborsClassifier(n_neighbors = 3)
knn3.fit(X_trainpca2,y_trianpca3)
Ypredpca3=knn3.predict(X_testpca3)
# Import and show confusion matrix
print(confusion_matrix(y_testpca3,Ypredpca3))
ConfusionMatrixDisplay.from_predictions(y_testpca3,Ypredpca3)