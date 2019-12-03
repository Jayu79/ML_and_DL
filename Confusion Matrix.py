from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np

class ConMatrix:
    def Matrix(y_test,predictions):
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(len(predictions)):
            if y_test[i]==predictions[i]==1:
                TP = TP+1
            if predictions[i]==1 and y_test[i]!=predictions[i]:
                FP = FP+1
            if y_test[i]==predictions[i]==0:
                TN = TN+1
            if predictions[i]==0 and y_test[i]!=predictions[i]:
                FN = FN+1
        print("The Confusion Matrix is")
        print(TP,"",FP)
        print(FN,"",TN)

    def precision(y_test, predictions):
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(len(predictions)):
            if y_test[i] == predictions[i] == 1:
                TP = TP + 1
            if predictions[i] == 1 and y_test[i] != predictions[i]:
                FP = FP + 1
            if y_test[i] == predictions[i] == 0:
                TN = TN + 1
            if predictions[i] == 0 and y_test[i] != predictions[i]:
                FN = FN + 1
        p = TP/(TP+FP)
        print("The precision is",p)
    def Recall(y_test, predictions):
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(len(predictions)):
            if y_test[i] == predictions[i] == 1:
                TP = TP + 1
            if predictions[i] == 1 and y_test[i] != predictions[i]:
                FP = FP + 1
            if y_test[i] == predictions[i] == 0:
                TN = TN + 1
            if predictions[i] == 0 and y_test[i] != predictions[i]:
                FN = FN + 1
        r = TP/(TP+FN)
        T = TP/(TP+FN)
        F =  FP/(FP+TN)
        print("The Recall is",r)
        print("The true positive rate is",T)
        print("The false positive rate is",F)
        plt.scatter(F,T,s=10,c='r')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()
        AUC = T*F
        print("The area under the curve is",AUC)
df = pd.read_csv("Two_points_KNN.csv")
new_data = df.rename(columns = {"9.434466063": "X", "-2.572000009":"Y", "0": "Class"})
X = new_data.drop(labels = ["Class"],axis=1)
Y = new_data["Class"].values
x,y = X,Y
x_train,x_test,y_train,y_test = train_test_split(x,y)
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
predictions = knn.predict(x_test)
ConMatrix.Matrix(y_test,predictions)
ConMatrix.precision(y_test,predictions)
ConMatrix.Recall(y_test,predictions)
