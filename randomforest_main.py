# LSTM for international airline passengers problem with regression framing
from random import randrange, random, seed

import numpy as np
from numpy import concatenate
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, average_precision_score, recall_score, f1_score
from math import sqrt
from matplotlib import pyplot
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# fix random seed for reproducibility

accuracy=0
def main():
  global accuracy
  np.random.seed(7)

# load the dataset
  dataframe_train = read_csv('train_data.csv', engine='python', sep=",", header=None)
  trainY = read_csv('train_label.csv', engine='python', header=None)
  dataset1 = dataframe_train.values
# 将整数值转换为浮点值，这些值更适合使用神经网络进行建模

  dataset1 = dataset1.astype('float32')
# normalize the dataset
# MinMaxScaler预处理类轻松地规范化数据集
  scaler1 = MinMaxScaler(feature_range=(0, 1))
  trainX = scaler1.fit_transform(dataset1)

  dataframe_test = read_csv('test_data.csv', engine='python', sep=",", header=None)
  testY = read_csv('test_label.csv', engine='python', header=None)
  dataset2 = dataframe_test.values
  dataset2 = dataset2.astype('float')
  scaler2 = MinMaxScaler(feature_range=(0, 1))
  testX = scaler2.fit_transform(dataset2)

  clf = RandomForestClassifier(n_estimators=10)  # 决策树的数目
  clf.fit(trainX,trainY)

  score = clf.score(testX, testY)
  result = clf.predict(testX)
# 准确率函数
  accuracy=accuracy_score(testY, result)
  print('准确率为：',accuracy)
# 精确率函数
  precision=average_precision_score(testY, result)
#print('精确率为：',precision)
# 召回率函数
  recall=recall_score(testY, result)
#print('召回率为：',recall)
# F1函数
#F1=f1_score(testY, result)
#print('F1为：',F1)

#plt.figure(num='随机森林对比')
#plt.plot(np.arange(len(result)), testY,'go-',label='true value')
#plt.plot(np.arange(len(result)),result,'ro-',label='predict value')
#plt.title('score: %f'%score)
#plt.legend()
#plt.show()


  np.savetxt('rf.csv',result, delimiter=',')

