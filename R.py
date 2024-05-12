'''
导入相关包
'''
import wfdb
import pywt
import seaborn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
import torch.utils.data as Data
from torch import nn
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import wfdb
import pywt
import seaborn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras import Model, Input
from sklearn.metrics import classification_report
from tensorflow.keras import Sequential

from tensorflow.keras.optimizers import SGD, Adam

from tensorflow.python.keras.layers.core import *
#from tensorflow.keras.utils import plot_model

# 测试集在数据集中所占的比例
RATIO = 0.2

# 小波去噪预处理
def denoise(data):
    # 小波变换
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    # 阈值去噪
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)
    # 小波反变换,获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata

# 读取心电数据和对应标签,并对数据进行小波去噪
def getDataSet(number, X_data, Y_data):
    ecgClassSet = ['N', 'A', 'V', 'L', 'R']
    # 读取心电数据记录
    print("正在读取 " + number + " 号心电数据...")
    # 读取MLII导联的数据
    record = wfdb.rdrecord('D:/ECG-Data/MIT-BIH-360/' + number, channel_names=['MLII'])
    data = record.p_signal.flatten()
    rdata = denoise(data=data)
    # 获取心电数据记录中R波的位置和对应的标签
    annotation = wfdb.rdann('D:/ECG-Data/MIT-BIH-360/' + number, 'atr')
    Rlocation = annotation.sample
    Rclass = annotation.symbol
    # 去掉前后的不稳定数据
    start = 10
    end = 5
    i = start
    j = len(annotation.symbol) - end
    # 因为只选择NAVLR五种心电类型,所以要选出该条记录中所需要的那些带有特定标签的数据,舍弃其余标签的点
    # X_data在R波前后截取长度为300的数据点
    # Y_data将NAVLR按顺序转换为01234
    while i < j:
        try:
            # Rclass[i] 是标签
            lable = ecgClassSet.index(Rclass[i])
            # 基于经验值，基于R峰向前取100个点，向后取200个点
            x_train = rdata[Rlocation[i] - 100:Rlocation[i] + 200]
            X_data.append(x_train)
            Y_data.append(lable)
            i += 1
        except ValueError:
            i += 1
    return

# 加载数据集并进行预处理
def loadData():
    numberSet = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
                 '116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208',
                 '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230',
                 '231', '232', '233', '234']
    dataSet = []
    lableSet = []
    for n in numberSet:
        getDataSet(n, dataSet, lableSet)
    # 转numpy数组,打乱顺序
    dataSet = np.array(dataSet).reshape(-1, 300)
    lableSet = np.array(lableSet).reshape(-1, 1)
    train_ds = np.hstack((dataSet, lableSet))
    np.random.shuffle(train_ds)
    # 数据集及其标签集
    X = train_ds[:, :300].reshape(-1, 300, 1)
    Y = train_ds[:, 300]
    # 测试集及其标签集
    shuffle_index = np.random.permutation(len(X))
    # 设定测试集的大小 RATIO是测试集在数据集中所占的比例
    test_length = int(RATIO * len(shuffle_index))
    # 测试集的长度
    test_index = shuffle_index[:test_length]
    # 训练集的长度
    train_index = shuffle_index[test_length:]
    X_test, Y_test = X[test_index], Y[test_index]
    X_train, Y_train = X[train_index], Y[train_index]
    return X_train, Y_train, X_test, Y_test

class buildModel():
    def __init__(self):
        super(buildModel, self).__init__()
        '''
        参数解释：(输入维度，隐藏层维度，网络层数)
        输入维度：每个x的输入大小，也就是每个x的特征数
        隐藏层：隐藏层的层数，若层数为1，隐层只有1层
        网络层数：网络层的大小
        '''
        self.rnn = nn.RNN(300, 50, 3, nonlinearity='tanh')
        self.linear = nn.Linear(50, 5)

    def forward(self, x):
        r_out, h_state = self.rnn(x)
        output = self.linear(r_out[:,-1,:])
        return output

def plotHeatMap(Y_test, Y_pred):
    con_mat = confusion_matrix(Y_test, Y_pred)
    # 绘图
    plt.figure(figsize=(4, 5))
    seaborn.heatmap(con_mat, annot=True, fmt='.20g', cmap='Blues')
    plt.ylim(0, 5)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()


def main():
    # 加载数据集并进行预处理
    X_train, Y_train, X_test, Y_test = loadData()
    print(X_train.shape)

    # 构建并编译模型

    model = buildModel()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # 训练模型
    model.fit(X_train, Y_train, epochs=30, batch_size=128, validation_split=RATIO)

    # 预测测试集
    Y_pred_probabilities = model.predict(X_test)
    Y_pred = np.argmax(Y_pred_probabilities, axis=1)  # 将概率最高的类别作为预测输出

    # 计算并打印每个类别的精确度、召回率和F1分数
    print(classification_report(Y_test, Y_pred, target_names=['N', 'A', 'V', 'L', 'R']))

    # 绘制混淆矩阵
    plotHeatMap(Y_test, Y_pred)



if __name__ == '__main__':
    main()