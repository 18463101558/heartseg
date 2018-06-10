
from keras.objectives import *
from keras.metrics import binary_crossentropy
import keras.backend as K
import tensorflow as tf
import numpy as np

# Softmax cross-entropy loss function for pascal voc segmentation
# and models which do not perform softmax.
# tensorlow only
def softmax_crossentropy(y_true, y_pred):
    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    #这里的k.int_shape()是返回预测值各个维度的大小张量，如（5,3,2），-1表明了是最后一个维度的值，
    # 而reshape（pred，（-1，size））是将pred变成（pred/size,size）,也就是变成了一个二维的张量
    log_softmax = tf.nn.log_softmax(y_pred)
    #计算softmax损失
    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
    #flatten可以将函数折叠成一维，注意是按行依次进行展开，
    #输入为n维的整数张量，形如(batch_size, dim1, dim2, ... dim(n-1))，最后一个参数是类的数量
    # 输出为(n+1)维的one-hot编码，形如(batch_size, dim1, dim2, ... dim(n-1), nb_classes)
    #最后输出一个二维结果
    unpacked = tf.unstack(y_true, axis=-1)
    #tf.unstack(),他妈的完全是脱了裤子放屁
    # 假如一个张量的形状是(A, B, C, D)。
    #如果axis == 0，则输出的张量是value[i, :, :, :],i取值为[0,A)，每个输出的张量的形状为(B,C,D)。
    y_true = tf.stack(unpacked[:-1], axis=-1)#：-1表示去除了最后一个维度
    #以指定的轴axis，将一个维度为R的张量数组转变成一个维度为R + 1的张量。即将一组张量以指定的轴，提高一个维度。
    #假设要转变的张量数组values的长度为N, 其中的每个张量的形状为(A, B, C)。
    #如果轴axis = 0，则转变后的张量的形状为(N, A, B, C)。
    #a = tf.constant([3,2,4,5,6])
    # b = tf.constant([1,6,7,8,0])
    # c = tf.stack([a,b],axis=0)
    #[[3 2 4 5 6][1 6 7 8 0]]，因为是默认按照axis等于1堆叠的
    #f = tf.unstack([a, b], axis=1)
    #[array([3, 1]), array([2, 6]), array([4, 7]), array([5, 8]), array([6, 0])]

    #最后返回依然是二维数组，是size*class数
    #
    cross_entropy = -K.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)#返回的是平均交叉熵

    return cross_entropy_mean


def dice_loss(y_true,y_pred):#计算dice指数（实际上这里是伪dice）
    nb_classes = K.int_shape(y_pred)[-1]  # 获取类别数量
    y_pred = K.reshape(y_pred, (-1, nb_classes))  # 压缩成二维,其实感觉好像也没啥必要来着
    y_pred= tf.nn.softmax( y_pred )
    y_true=K.one_hot(tf.to_int32(K.flatten(y_true)),
                       nb_classes)
    fenzi=K.mean(2*tf.to_float(np.multiply(y_pred,y_true)))
    fenmu=K.mean(tf.to_float(y_pred))+K.mean(tf.to_float(y_true))
    return 1-fenzi/fenmu
