#!usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    """载入数据及标准化"""
    print("load_data:")
    data_positive = np.loadtxt("F:/projectfile/ranking/data/Test_Lux_ZZ.csv", delimiter=",")
    data_nagetive = np.loadtxt("F:/projectfile/ranking/data/Test_Lux_WW.csv", delimiter=",")
    data = np.concatenate((data_positive, data_nagetive), axis=0)
    X = data[:, 1:];y = data[:, 0];labels=[]
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    for i in range(len(y)):
        if y[i]==0:
            labels.append([1,0])
        else:
            labels.append([0,1])
    index = np.loadtxt("F:/projectfile/ranking\data/ranking.txt")
    index0=list(map(int,index[0]))
    return X[:,index0],np.array(labels)

def build_network():
    """构建MLP网络"""
    samples=tf.placeholder("float",[None,37])
    y=tf.placeholder("float",[None,2])

    fc1=tf.layers.dense(inputs=samples,units=512,activation=tf.nn.relu,name="fc1")
    fc2=tf.layers.dense(inputs=fc1,units=256,activation=tf.nn.relu,name="fc2")
    fc3 = tf.layers.dense(inputs=fc2, units=128, activation=tf.nn.relu,name="fc3")
    out = tf.layers.dense(inputs=fc3, units=2, activation=None,name="out")
    out2 = tf.nn.softmax(out)

    """计算loss，进行优化"""
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=out))
    #loss = tf.reduce_mean(-tf.reduce_sum(y* tf.log(out),reduction_indices=[1]))
    optimizer=tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(loss)

    """评估正确率"""
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(out2,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        X0,y0=load_data()
        print("bulid_network:")

        """划分训练集和测试集"""
        X_train, X_test, y_train, y_test = train_test_split(X0, y0, test_size=0.33, random_state=42)
        steps=X_train.shape[0]//64

        for epoch in range(600):
            for step in range(steps):
                batchx=X_train[step*64:(step+1)*64,:]
                batchy=y_train[step*64:(step+1)*64,:]
                train_loss,_=sess.run([loss,optimizer],feed_dict={samples:batchx,y:batchy})

                """计算在测试集上的准确率"""
                if step==50:
                    acc = sess.run(accuracy, feed_dict={samples: X_test, y: y_test})
                    # print("{}/{}:{}",format(epoch,step,train_loss))
                    print(epoch, step, train_loss,acc)

build_network()






