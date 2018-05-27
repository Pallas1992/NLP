# -*- coding: utf-8 -*-

"""
@auther: Tie Ruixue
@file: Svm.py
@time: 2018/5/27 15:41

"""
from sklearn import svm
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
from sklearn.grid_search import GridSearchCV

iris = load_iris()
data = iris["data"]
label = iris["target"]
def show_accuracy(list1, list2, str):
    n = 0
    for i in range(len(list1)):
        if list1[i]==list2[i]:
            n +=1
    accuracy = n/len(list1)*1.0
    return accuracy

x = data
y = label
x_train, x_test, y_train, y_test = train_test_split(x, y,train_size=0.6,random_state=1)
"""
train_data：所要划分的样本特征集

　　train_target：所要划分的样本结果

　　test_size：样本占比，如果是整数的话就是样本的数量

　　random_state：是随机数的种子。

　　随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。随机数的产生取决于种子，随机数和种子之间的关系遵从以下两个规则：种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。
"""
#调优
# parameters = {'C':[0.1,0.003,0.5,0.009,0.01,0.04,0.08,1],
#               'kernel':('linear','rbf',),
#               'gamma':[20,0.005,0.1,0.15,0.20,0.23,10],
#               'decision_function_shape':['ovr']
#              }
parameters = {'C':[0.1,0.5,0.01,0.08,1],
              'kernel':('linear','rbf',),
              'gamma':[0.1,15,0.20,10],
              'decision_function_shape':['ovr']
             }
#GridSearchCV，sklearn的自动调优函数
svr = svm.SVC()
clf = GridSearchCV(svr, parameters)
# clf = svm.SVC(C=2, kernel='rbf', gamma=20, decision_function_shape='ovr')#不进行调优
clf.fit(x_train, y_train)
#使用a储存调优后的参数结果
a=pd.DataFrame(clf.grid_scores_)
print(a)

#按照mean_test_score降序排列
a.sort_values(['mean_validation_score'],ascending=False)

#输出最好的分类器参数，以及测试集的平均分类正确率
print(clf.best_estimator_)
# print(clf.best_score_)

"""
　kernel='linear'时，为线性核，C越大分类效果越好，但有可能会过拟合（defaul C=1）。

　　 kernel='rbf'时（default），为高斯核，gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合。

　　decision_function_shape='ovr'时，为one v rest，即一个类别与其他类别进行划分，

　　decision_function_shape='ovo'时，为one v one，即将类别两两之间进行划分，用二分类的方法模拟多分类的结果。
"""
print (clf.score(x_train, y_train))  # 精度
y_train_pre = clf.predict(x_train)
# print('训练集: % f'% show_accuracy(y_hat, y_train,"训练集"))
print(clf.score(x_test, y_test))
y_hat = clf.predict(x_test)
# print(show_accuracy(y_hat, y_test, '测试集'))
# print ('decision_function:\n', clf.decision_function(x_train))
# print ('\npredict:\n', clf.predict(x_train))