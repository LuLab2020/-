import pandas as pd
import csv
from numpy import concatenate
def app(f1, f2, f3, f4):
    dataset = pd.concat((f1, f2, f3), axis=1, ignore_index=True)
    data = pd.DataFrame(dataset)
    data.to_csv(f4, header=None, index=None)

f1 = pd.read_csv('bow_train1.csv', engine='python', sep=" ", header=None)
f2 = pd.read_csv('bow_train2.csv', engine='python', sep=" ", header=None)
f3 = pd.read_csv('train_data3.csv', engine='python', sep=" ", header=None)
f4 = "train_data.csv"

f5 = pd.read_csv('bow_test1.csv', engine='python', sep=" ", header=None)
f6 = pd.read_csv('bow_test2.csv', engine='python', sep=" ", header=None)
f7 = pd.read_csv('test_data3.csv', engine='python', sep=" ", header=None)
f8 = "test_data.csv"
#
# f9 = pd.read_csv('bow_val1.csv', engine='python', sep=" ", header=None)
# f10 = pd.read_csv('bow_val2.csv', engine='python', sep=" ", header=None)
# f11 = pd.read_csv('val_data3.csv', engine='python', sep=" ", header=None)
# f12 = "val_data.csv"

app(f1, f2, f3, f4)
app(f5, f6, f7, f8)
# app(f9, f10, f11, f12)