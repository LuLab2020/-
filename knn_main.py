# LSTM for international airline passengers problem with regression framing
import numpy as np
from numpy import concatenate
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
from sklearn import svm, neighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, average_precision_score, recall_score, f1_score
from math import sqrt

from sklearn.metrics import accuracy_score


# fix random seed for reproducibility
# 在我们做任何事情之前，最好修复随机数种子以确保我们的结果可重复


import csv
import random
import math
import operator
accuracy=0
def main():
  global accuracy

  np.random.seed(7)


  dataframe_train = read_csv('train_data.csv', engine='python', sep=",", header=None)
  trainY = read_csv('train_label.csv', engine='python', header=None)
  dataset1 = dataframe_train.values
# 将整数值转换为浮点值，这些值更适合使用神经网络进行建模
# dataset1=float(dataset1)
  dataset1 = dataset1.astype('float32')
# normalize the dataset
# MinMaxScaler预处理类轻松地规范化数据集
  scaler1 = MinMaxScaler(feature_range=(0, 1))
  trainX = scaler1.fit_transform(dataset1)

  dataframe_test = read_csv('test_data.csv', engine='python', sep=",", header=None)
  testY = read_csv('test_label.csv', engine='python', header=None)
  dataset2 = dataframe_test.values
  dataset2 = dataset2.astype('float')
# dataset2=float(dataset2)
  scaler2 = MinMaxScaler(feature_range=(0, 1))
  testX = scaler2.fit_transform(dataset2)
#
# dataframe_val = read_csv('val_data.csv', engine='python', sep=",", header=None)
# val_y = read_csv('val_label.csv', engine='python', header=None)
# dataset3 = dataframe_val.values
# dataset3 = dataset3.astype('float')
# # dataset3=float(dataset3)
# scaler3 = MinMaxScaler(feature_range=(0, 1))
# val_set = scaler3.fit_transform(dataset3)
# val = concatenate((val_set[:, :], val_y), axis=1)

#best_score=0.0  #先设置一个精确度的初始值
#best_k=-1         #设置一个k的初始值
#for k in range(1,20):
#    knn_clf=neighbors.KNeighborsRegressor(n_neighbors=k)
#    knn_clf.fit(trainX, trainY)
#    score=knn_clf.score(testX, testY)
#    if score > best_score:
#        best_score=score
#        best_k=k

#print('best_score=%s'%(best_score))
#print('best_k=%s'%(best_k))

  model = neighbors.KNeighborsRegressor(n_neighbors=20)

  model.fit(trainX, trainY)  # fit the model

  score = model.score(testX, testY)
  result = model.predict(testX)
  result = [int(item>0.1) for  item in result]

# 准确率函数
  accuracy=accuracy_score(testY, result)
  print('准确率为：',accuracy)
# 精确率函数
  precision=average_precision_score(testY, result)
  print('精确率为：',precision)
# 召回率函数
  recall=recall_score(testY, result)
  print('召回率为：',recall)
# F1函数
  F1=f1_score(testY,result)
  print('F1为：',F1)



  error = sqrt(mean_squared_error(testY, result))  # calculate rmse

  print('RMSE value for is:', error)
  np.savetxt('knn.csv', result, delimiter=',')
