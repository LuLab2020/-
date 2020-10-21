# coding=utf-8

import csv
import time
import datetime


f1 = csv.reader(open('cicDDostrain.csv', 'r'))
f2 = open('train_data1.txt', 'w')
f3 = open('train_data2.txt', 'w')
f4 = open('train_data3.csv', 'w')
f5 = open('train_label.csv', 'w')
f6 = csv.reader(open('cicBobtest.csv', 'r'))
f7 = open('test_data1.txt', 'w')
f8 = open('test_data2.txt', 'w')
f9 = open('test_data3.csv', 'w')
f10 = open('test_label.csv', 'w')
def normalize(f, f_data1, f_data2, f_data3, f_label):
    j=0
    for data in f:
        j+=1
        if j==32500:
            break
        ip_s = data[0].split(".")
        ip_src = ""
        for i in ip_s:
            i = i.rjust(3, "0")
            ip_src += i

        ip_d = data[1].split(".")
        ip_dst = ""
        for i in ip_d:
            i = i.rjust(3, "0")
            ip_dst += i

        t = time.strptime(data[5], "%m/%d/%Y %H:%M")#根据指定的格式把时间字符串解析为时间元组
        t = int(time.mktime(t))#返回用秒数来返回时间的浮点数
        #t = data[5].split(".")
        #ts = t[0][5:]+t[1][0:3]
        ts = t
        t = int(ts) / pow(10, 7)

        sport = int(data[3]) / pow(10, len(data[3])-1)

        dport = int(data[4]) / pow(10, len(data[4]) - 1)

        d_t = data[6]#.split(":")#生存时间
        #delta_t = str(int(d_t[0])*3600+int(d_t[1])*60+int(d_t[2]))
        d_t = int(d_t) / pow(10, len(d_t) - 1)

        s1 = str(ip_src) + "\n"
        s2 = str(ip_dst) + "\n"
        s3 = data[2] + " " + str(sport) + " " + str(dport) + " " + str(t) + \
             " " + str(d_t) + " " + data[7] + " " + data[8] + " " + data[9] + "\n"
        f_data1.writelines(s1)
        f_data2.writelines(s2)
        f_data3.writelines(s3)
        if (data[10] == 'BENIGN'):
         flag = '0'
        else:
         flag = '1'
        f_label.writelines(flag+ "\n")
    f_data1.close()
    f_data2.close()
    f_data3.close()
    f_label.close()


# f11 = csv.reader(open('val_sorted.csv', 'r'))
# f12 = open('val_data1.txt', 'w')
# f13 = open('val_data2.txt', 'w')
# f14 = open('val_data3.csv', 'w')
# f15 = open('val_label.csv', 'w')
normalize(f1,f2,f3,f4,f5)
normalize(f6,f7,f8,f9,f10)
# normalize(f11,f12,f13,f14,f15)