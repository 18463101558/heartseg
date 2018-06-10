# coding=utf-8
import keras.backend as K
import tensorflow as tf
from tensorflow.contrib.metrics import streaming_mean_iou
import  numpy as np
"""
def accuracy(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]#获取类别数量
    y_pred = K.reshape(y_pred, (-1, nb_classes))#压缩成二维，方便计算softmax

    legal_labels=K.ones_like(tf.to_int32(K.flatten(y_true)), name=None )
    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)),
                       nb_classes)
    #准确率是看两个标签正好一致的位置，例如pred在第7个位置而标签值也正好此时为7
    #分母是总的像素点数量
    return K.sum(tf.to_float( K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))) / K.sum(tf.to_float(legal_labels))"""

def FDM(y_true, y_pred):#肺动脉
    nb_classes = y_pred.shape[-1]  # 获取类别数量
    #y_pred = K.reshape(y_pred, (-1, nb_classes))#压缩成二维，方便计算softmax
    y_pred = np.argmax(y_pred, axis=-1).astype(np.uint8)

    y_pred = K.one_hot(tf.to_int32(K.flatten(y_pred)),nb_classes)
    unpacked = tf.unstack(y_pred, axis=-1)
    y_pred = unpacked[-1]#仅仅取出倒数第二个维度


    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)),nb_classes)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = unpacked[-1]#仅仅取出最后一个维度


    fenzi=y_pred*y_true
    fenzi=K.mean(tf.to_float(K.flatten(fenzi)))
    fenmu1=K.mean(tf.to_float(K.flatten(y_pred)))
    fenmu2=K.mean(tf.to_float(K.flatten(y_true)))
    fenmu=fenmu1+fenmu2#分母1远远小于分子啊

    standard = tf.constant(value = 0.00001, dtype = tf.float32) #其实s是用于判断分母是否为0

    result = tf.cond(tf.less(fenmu,standard), lambda: tf.constant(value = 1.0, dtype = tf.float32), lambda: (2*fenzi)/(fenmu1+fenmu2))
    return result

def SZDM(y_true, y_pred):#升主动脉
    nb_classes = y_pred.shape[-1]#获取类别数量

    y_pred = np.argmax(y_pred, axis=-1).astype(np.uint8)


    y_pred = K.one_hot(tf.to_int32(y_pred),nb_classes)
    unpacked = tf.unstack(y_pred, axis=-1)
    y_pred = unpacked[-2]#仅仅取出倒数第二个维度

    y_true = K.one_hot(tf.to_int32(y_true),nb_classes)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = unpacked[-2]#仅仅取出最后一个维度

    pred=K.flatten(y_pred)
    true=K.flatten(y_true)
    IOU=pred*true
    IOU=K.reshape(IOU, (-1,512))

    fenzi=K.sum(tf.to_float(K.flatten(IOU)))
    fenmu1=K.sum(tf.to_float(pred))
    fenmu2=K.sum(tf.to_float(true))
    fenmu=fenmu1+fenmu2#分母1远远小于分子啊
    standard = tf.constant(value = 0.00001, dtype = tf.float32) #其实s是用于判断分母是否为0
    result = tf.cond(tf.less(fenmu,standard), lambda: tf.constant(value = 1.0, dtype = tf.float32), lambda: (2*fenzi)/(fenmu1+fenmu2))
    return result

def YXSXQ(y_true, y_pred):#右心室血腔
    nb_classes = y_pred.shape[-1]#获取类别数量

    y_pred = np.argmax(y_pred, axis=-1).astype(np.uint8)


    y_pred = K.one_hot(tf.to_int32(y_pred),nb_classes)
    unpacked = tf.unstack(y_pred, axis=-1)
    y_pred = unpacked[-3]#仅仅取出倒数第二个维度

    y_true = K.one_hot(tf.to_int32(y_true),nb_classes)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = unpacked[-3]#仅仅取出最后一个维度

    pred=K.flatten(y_pred)
    true=K.flatten(y_true)
    IOU=pred*true
    IOU=K.reshape(IOU, (-1,512))

    fenzi=K.sum(tf.to_float(K.flatten(IOU)))
    fenmu1=K.sum(tf.to_float(pred))
    fenmu2=K.sum(tf.to_float(true))
    fenmu=fenmu1+fenmu2#分母1远远小于分子啊
    standard = tf.constant(value = 0.00001, dtype = tf.float32) #其实s是用于判断分母是否为0
    result = tf.cond(tf.less(fenmu,standard), lambda: tf.constant(value = 1.0, dtype = tf.float32), lambda: (2*fenzi)/(fenmu1+fenmu2))
    return result

def YXFXQ(y_true, y_pred):#右心房血腔
    nb_classes = y_pred.shape[-1]#获取类别数量

    y_pred = np.argmax(y_pred, axis=-1).astype(np.uint8)


    y_pred = K.one_hot(tf.to_int32(y_pred),nb_classes)
    unpacked = tf.unstack(y_pred, axis=-1)
    y_pred = unpacked[-4]#仅仅取出倒数第二个维度

    y_true = K.one_hot(tf.to_int32(y_true),nb_classes)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = unpacked[-4]#仅仅取出最后一个维度

    pred=K.flatten(y_pred)
    true=K.flatten(y_true)
    IOU=pred*true
    IOU=K.reshape(IOU, (-1,512))

    fenzi=K.sum(tf.to_float(K.flatten(IOU)))
    fenmu1=K.sum(tf.to_float(pred))
    fenmu2=K.sum(tf.to_float(true))
    fenmu=fenmu1+fenmu2#分母1远远小于分子啊
    standard = tf.constant(value = 0.00001, dtype = tf.float32) #其实s是用于判断分母是否为0
    result = tf.cond(tf.less(fenmu,standard), lambda: tf.constant(value = 1.0, dtype = tf.float32), lambda: (2*fenzi)/(fenmu1+fenmu2))
    return result

def ZXSXQ(y_true, y_pred):#左心室血腔
    nb_classes = y_pred.shape[-1]#获取类别数量

    y_pred = np.argmax(y_pred, axis=-1).astype(np.uint8)


    y_pred = K.one_hot(tf.to_int32(y_pred),nb_classes)
    unpacked = tf.unstack(y_pred, axis=-1)
    y_pred = unpacked[-5]#仅仅取出倒数第二个维度

    y_true = K.one_hot(tf.to_int32(y_true),nb_classes)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = unpacked[-5]#仅仅取出最后一个维度

    pred=K.flatten(y_pred)
    true=K.flatten(y_true)
    IOU=pred*true
    IOU=K.reshape(IOU, (-1,512))
    fenzi=K.sum(tf.to_float(K.flatten(IOU)))
    fenmu1=K.sum(tf.to_float(pred))
    fenmu2=K.sum(tf.to_float(true))
    fenmu=fenmu1+fenmu2#分母1远远小于分子啊
    standard = tf.constant(value = 0.00001, dtype = tf.float32) #其实s是用于判断分母是否为0
    result = tf.cond(tf.less(fenmu,standard), lambda: tf.constant(value = 1.0, dtype = tf.float32), lambda: (2*fenzi)/(fenmu1+fenmu2))
    return result

def ZXFXQ(y_true, y_pred):#左心房血腔
    nb_classes = y_pred.shape[-1]#获取类别数量

    y_pred = np.argmax(y_pred, axis=-1).astype(np.uint8)


    y_pred = K.one_hot(tf.to_int32(y_pred),nb_classes)
    unpacked = tf.unstack(y_pred, axis=-1)
    y_pred = unpacked[-6]#仅仅取出倒数第二个维度

    y_true = K.one_hot(tf.to_int32(y_true),nb_classes)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = unpacked[-6]#仅仅取出最后一个维度

    pred=K.flatten(y_pred)
    true=K.flatten(y_true)
    IOU=pred*true
    IOU=K.reshape(IOU, (-1,512))

    fenzi=K.sum(tf.to_float(K.flatten(IOU)))
    fenmu1=K.sum(tf.to_float(pred))
    fenmu2=K.sum(tf.to_float(true))
    fenmu=fenmu1+fenmu2#分母1远远小于分子啊
    standard = tf.constant(value = 0.00001, dtype = tf.float32) #其实s是用于判断分母是否为0
    result = tf.cond(tf.less(fenmu,standard), lambda: tf.constant(value = 1.0, dtype = tf.float32), lambda: (2*fenzi)/(fenmu1+fenmu2))
    return result

def ZXSXJ(y_true, y_pred):#左心室心肌
    nb_classes = y_pred.shape[-1]#获取类别数量

    y_pred = np.argmax(y_pred, axis=-1).astype(np.uint8)


    y_pred = K.one_hot(tf.to_int32(y_pred),nb_classes)
    unpacked = tf.unstack(y_pred, axis=-1)
    y_pred = unpacked[-7]#仅仅取出倒数第二个维度

    y_true = K.one_hot(tf.to_int32(y_true),nb_classes)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = unpacked[-7]#仅仅取出最后一个维度

    pred=K.flatten(y_pred)
    true=K.flatten(y_true)
    IOU=pred*true
    IOU=K.reshape(IOU, (-1,512))

    fenzi=K.sum(tf.to_float(K.flatten(IOU)))
    fenmu1=K.sum(tf.to_float(pred))
    fenmu2=K.sum(tf.to_float(true))
    fenmu=fenmu1+fenmu2#分母1远远小于分子啊
    standard = tf.constant(value = 0.00001, dtype = tf.float32) #其实s是用于判断分母是否为0
    result = tf.cond(tf.less(fenmu,standard), lambda: tf.constant(value = 1.0, dtype = tf.float32), lambda: (2*fenzi)/(fenmu1+fenmu2))
    return result
def average(y_true, y_pred):
    feidongmai=FDM(y_true, y_pred)
    shengzhudongmai=SZDM(y_true, y_pred)
    youxinshixueqiang=YXSXQ(y_true, y_pred)
    youxinfangxueqiang=YXFXQ(y_true, y_pred)
    zuoxinshixueqiang=ZXSXQ(y_true, y_pred)
    zuoxinfangxueqiang=ZXFXQ(y_true, y_pred)
    zuoxinshixinji=ZXSXJ(y_true, y_pred)
    average=feidongmai+shengzhudongmai+youxinshixueqiang+youxinfangxueqiang+zuoxinshixueqiang+zuoxinfangxueqiang+zuoxinshixinji
    return average/7
"""
def dice_acc(y_true,y_pred):#计算dice指数（实际上这里是伪dice）
    #dice指数是看二者同时为1的情况，也就是在one-hot编码中，二者预测点正好相同的个数

    nb_classes = K.int_shape(y_pred)[-1]#获取类别数量
    y_pred = K.reshape(y_pred, (-1, nb_classes))#压缩成二维，方便计算softmax
    y_pred = tf.nn.softmax(y_pred)#转变成概率输出
    y_pred = np.argmax(y_pred, axis=-1).astype(np.uint8)
    y_pred=K.one_hot(tf.to_int32(K.flatten(y_pred)),nb_classes)

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)),
                   nb_classes)

    fenzi=K.minimum(y_pred, y_true)
    fenzi=K.sum(tf.to_float(K.flatten(fenzi)))
    fenmu1=K.sum(tf.to_float(K.flatten(y_pred)))
    fenmu2=K.sum(tf.to_float(K.flatten(y_true)))
    return fenzi/(fenmu2+fenmu1)
"""