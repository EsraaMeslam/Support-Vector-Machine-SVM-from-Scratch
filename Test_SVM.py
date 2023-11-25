from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

import SVM as SVM


X,y=datasets .make_blobs(n_samples=50,n_features=2,centers=2,cluster_std=1.05,random_state=40)

y=np.where(y==0,1,-1)

x_train ,x_test,y_train,y_test=train_test_split(X,y ,test_size=.2,random_state=4)


model=SVM()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)









def accuracy(y_test,y_pred):
    acc=np.sum(y_test==y_pred)/len(y_test)
    return acc


print("Accurcy",accuracy(y_test,y_pred))
