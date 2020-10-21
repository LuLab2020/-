# coding=utf-8
# 上面是因为worker计算内容各不相同，不过在深度学习中，一般每个worker的计算内容都是一样的，
# 都是计算神经网络的每个batch前向传导，所以一般代码是重用的
import tensorflow.compat.v1 as tf
import os

from opt_einsum.backends import torch

import randomforest_main
# 现在假设我们有A、B台机器，首先需要在各台机器上写一份代码，并跑起来，各机器上的代码内容大部分相同
# 除了开始定义的时候，需要各自指定该台机器的task之外。以机器A为例子，A机器上的代码如下：

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
tf.disable_eager_execution()
cluster = tf.train.ClusterSpec({
    "worker": [
        "192.168.31.88:2222",
        "192.168.31.110:2224", # 格式 IP地址：端口号，第一台机器A的IP地址 ,在代码中需要用这台机器计算的时候，就要定义：/job:worker/task:0
    ],
    "ps": [
        "192.168.31.88:2223"  # 第四台机器的IP地址 对应到代码块：/job:ps/task:0
    ]})

# 不同的机器，下面这一行代码各不相同，server可以根据job_name、task_index两个参数，查找到集群cluster中对应的机器
isps = True
if isps:
    server = tf.train.Server(cluster, job_name='ps', task_index=0)  # 找到‘worker’名字下的，task0，也就是机器A
    server.join()
else:
    server = tf.train.Server(cluster, job_name='worker', task_index=0)  # 找到‘worker’名字下的，task0，也就是机器A
    with tf.device(tf.train.replica_device_setter(worker_device='/job:worker/task:0', cluster=cluster)):
        os.system('python3 extract_feature_1.py')
        os.system('python3 bow_2.py')
        os.system('python3 data_append_3.py')
        randomforest_main.main()
        acc=tf.constant([randomforest_main.accuracy])

        summary_op = tf.summary.merge_all()
        init_op = tf.initialize_all_variables()
        sv = tf.train.Supervisor(init_op=init_op, summary_op=summary_op)
        with sv.managed_session(server.target) as sess:
            print(sess.run(acc))
