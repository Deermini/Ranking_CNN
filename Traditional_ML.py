#!usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import collections
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def classifer():
    """载入数据及标准化"""
    data_positive=np.loadtxt("F:/projectfile/ranking/data/Test_Lux_ZZ.csv",delimiter=",")
    data_nagetive=np.loadtxt("F:/projectfile/ranking/data/Test_Lux_WW.csv",delimiter=",")
    data=np.concatenate((data_positive,data_nagetive),axis=0)
    X=data[:,1:];y=data[:,0]
    # scaler=StandardScaler().fit(X)
    # X=scaler.transform(X)
    # index=np.loadtxt("F:/projectfile/ranking/data/ranking.txt")

    """划分训练集和测试集"""
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.10,random_state=42)
    print(collections.Counter(y_test))

    """构建分类器"""
    clf=GradientBoostingClassifier(n_estimators=300,max_depth=6)
    #clf=RandomForestClassifier(n_estimators=200)
    clf.fit(X_train,y_train)
    prediction=clf.predict(X_test)
    acc=accuracy_score(y_test,prediction)
    print("accuracy:",acc)
    with open("ML_result.txt",'a') as f:
        f.write("\n"+str(clf)+"\n")
        f.write("acuracy:"+str(acc)+"\n")

if __name__=='__main__':
    classifer()




