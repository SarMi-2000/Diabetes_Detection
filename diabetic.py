import numpy as np
import pandas as pd
data=pd.read_csv("Diabetes.csv")
data.info()
data.isnull.sum()
X=data.iloc[:,:-1]
y=data.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_text,y_train,y_text=train_test_split(X,y,random_state=0,test_size=25)
from sklearn.ensemble import RandomForestClassifier
Classfier=RandomForestClassifier(n_estimators=8,criterion='entropy',random_state=0)
Classfier.fit(X_train,y_train)
y_pred=Classfier.predict(X_text)
from sklearn.metrics import accuracy_score
acc_randomf=round(accuracy_score(y_pred,y_text),2)*100
print("Accuracy Random Forest Classifier: ",acc_randomf)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,r2_score,classification_report
logreg=LogisticRegression(solver='lbfgs',max_iter=1000)
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_text)
acc_logreg=round(accuracy_score(y_pred,y_text),2)*100
print("Accuracy Logistic Regression: ",acc_logreg)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
acc_knn=round(accuracy_score(y_pred,y_text),2)*100
print("Accuracy K Neighbors Classifier: ",acc_knn)