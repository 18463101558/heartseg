# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
import pickle
from keras.optimizers import SGD, Adam, Nadam
from keras.callbacks import *
from keras.objectives import *
from keras.metrics import binary_accuracy
from keras.models import load_model
import keras.backend as K
import keras
from LossHistory import *
from models import *
from utils.loss_function import *
from utils.metrics import *
from utils.SegDataGenerator import *
import time

def draw_structor_to_png(save_path,model):
    img_path = os.path.join(save_path, "model.png")
    #这一个为了绘制出模型图案

    #vis_util.plot(model, to_file=save_path, show_shapes=True)


def train(batch_size, epochs, lr_base, lr_power, weight_decay, classes,
          model_name, train_file_path, val_file_path,
          data_dir, label_dir, target_size=None, batchnorm_momentum=0.9,
          loss_fn=softmax_crossentropy,
          metrics=[],
          label_cval=255):

    #设置批文件大小
    if target_size:
        input_shape = target_size + (1,)#之前给的是320*320
    else:
        input_shape = (None, None, 1)
    batch_shape = (batch_size,) + input_shape#每一个batch的大小



    # ###############设置学习率随权重衰减####################
    def lr_scheduler(epoch, mode='power_decay'):#设置学习率参数，注意默认参数是power_decay

        if mode is 'power_decay':
            # ** 在python里面表示幂运算,radius**3 表示radius的3次方
            lr = lr_base * ((1 - float(epoch)/epochs) ** lr_power)
        if mode is 'exp_decay':
            # exponential decay
            lr = (float(lr_base) ** float(lr_power)) ** float(epoch+1)
        # adam default lr
        if mode is 'adam':
            lr = 0.001

        if mode is 'progressive_drops':
            # drops as progression proceeds, good for sgd
            if epoch > 0.9 * epochs:
                lr = 0.0001
            elif epoch > 0.75 * epochs:
                lr = 0.001
            elif epoch > 0.5 * epochs:
                lr = 0.01
            else:
                lr = 0.1

        print('回调一次：lr: %f' % lr)
        return lr
    scheduler = LearningRateScheduler(lr_scheduler)#设置学习率



    #生成模型
    model = globals()[model_name](weight_decay=weight_decay,#正则化项目的系数
                                  input_shape=input_shape,#每一张输入图片的大小
                                  batch_momentum=batchnorm_momentum,#动量前面的系数
                                  classes=classes)#glob是为了访问全局变量，并且修改这个model的参数

    # 定义优化器，并且进行编译,这里使用带动量的SGD下降
    optimizer = SGD(lr=lr_base, momentum=0.9)
    # optimizer = Nadam(lr=lr_base, beta_1 = 0.825, beta_2 = 0.99685)
    """----------------------loss_fn用法---------------------------"""
    # 可以通过传递预定义目标函数名字指定目标函数,也可以传递一个Theano/TensroFlow的符号函数作为目标函数，
    # 该函数对每个数据点应该只返回一个标量值，并以下列两个参数为参数
    #y_true：真实的数据标签，Theano/TensorFlow张量
    #y_pred：预测值，与y_true相同shape的Theano/TensorFlow张量
    """----------------------metrics用法---------------------------"""
    #性能评估模块提供了一系列用于模型性能评估的函数, 这些函数在模型编译时由metrics关键字设置
    #性能评估函数类似与目标函数, 只不过请注意该性能的评估结果讲不会用于训练.
    #可以通过字符串来使用域定义的性能评估函数
    #y_true:真实标签,theano/tensorflow张量
    #y_pred:预测值, 与y_true形式相同的theano/tensorflow张量
    model.compile(loss=loss_fn,
                  optimizer=optimizer,
                  metrics=metrics)

    # 设置model的保存路径
    current_dir = os.path.dirname(os.path.realpath(__file__))  # 获得当前py文件所在的路径
    save_path = os.path.join(current_dir, 'Models/' + model_name)  # 给出文件model保存的路径
    checkpoint_path = os.path.join(save_path, 'checkpoint_weights.hdf5')
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)  # /home/ye/桌面/我的tensorflow之旅/heart/Models/AtrousFCN_Resnet50_16s
    #model.load_weights(checkpoint_path, by_name=True)#重新训练

    # 保存model的结构
    model_path = os.path.join(save_path, "model.json")
    f = open(model_path, 'w')
    model_json = model.to_json()
    f.write(model_json)
    f.close#将model的结构写入到文件里面
    #draw_structor_to_png(save_path,model)#这一个是将网络结构图画出来的，老是装不上，弃疗了
    callbacks = [scheduler]#设置学习率

    # ####################### 保存结果进行可视化 ###########################
    if K.backend() == 'tensorflow':
        tensorboard = TensorBoard(log_dir=os.path.join(save_path, 'logs'), write_graph=True)
        callbacks.append(tensorboard)
    #用于可视化

    # ################### 每个epoch的结尾处，尝试一次保留权重#######################
    checkpoint = ModelCheckpoint(filepath=os.path.join(save_path, 'checkpoint_weights.hdf5'), save_weights_only=True)
    callbacks.append(checkpoint)
    ##################################################################################################
    # losshistory = LossHistory()
    # callbacks.append(losshistory)
    ###################################################################################################
    # set data generator and train
    train_datagen = SegDataGenerator()#初始化一些变量
    val_datagen = SegDataGenerator()#获取验证集并且不剪取图片

    #报错来自于这个地方，因为它找不到train.path
    def get_file_len(file_path):
        fp = open(file_path)
        lines = fp.readlines()
        fp.close()
        return len(lines)


    steps_per_epoch = int(np.ceil(get_file_len(train_file_path) / float(batch_size)))
    # 发生器产生的步骤总数（样品批次）。它通常应该等于数据集中样本的数量除以批量大小。

    #第一个参数是生成函数

    #generator：生成器函数，生成器的输出应该为：
    # 一个形如（inputs，targets）的tuple
    #一个形如（inputs, targets, sample_weight）的tuple。所有的返回值都应该包含相同数目的样本。生成器将无限在数据集上循环。每个epoch以经过模型的样本数达到samples_per_epoch时，记一个epoch结束
    history = model.fit_generator(
        generator=train_datagen.flow_from_directory(
            file_path=train_file_path,
            data_dir=data_dir,
            label_dir=label_dir,
            classes=classes,
            target_size=target_size, color_mode='grayscale',
            batch_size=1, shuffle=True,
        ),
        steps_per_epoch=steps_per_epoch,#当生成器返回steps_per_epoch次数据时计一个epoch结束，执行下一个epoch
        epochs=epochs,#整数，数据迭代的轮数
        callbacks=callbacks,
        workers=4,#最大进程数量
        validation_data=val_datagen.flow_from_directory(
            file_path=val_file_path,
            data_dir=data_dir,
            label_dir=label_dir,
            classes=classes,
            target_size=target_size, color_mode='grayscale',
            batch_size=1, shuffle=True,
        ),
       )

    model.save_weights(save_path+'/model.hdf5')

if __name__ == '__main__':
    model_name = 'AtrousFCN_Resnet50_16s'
    batch_size =1
    batchnorm_momentum = 0.95
    epochs = 250
    lr_base = 0.01 * (float(batch_size) / 16)
    lr_power = 0.9
    resume_training = False
    weight_decay = 0.0001/2
    target_size = (512, 512)#图片大小

    # 设置加载文件路径啥的
    train_file_path = os.path.expanduser('Segmentation/train.txt') #Data/VOClarge/VOC2012/ImageSets/Segmentation

    val_file_path   = os.path.expanduser('Segmentation/val.txt')
    data_dir        = os.path.expanduser('train')
    label_dir       = os.path.expanduser('label')
    data_suffix=''
    label_suffix=''
    #classes = 21
    # #pascal数据集加上背景在内一共是21类
    classes =8

    # ###################### loss function & metric ########################
    loss_fn = dice_loss
    metrics =[average,FDM,SZDM,YXSXQ,YXFXQ,ZXSXQ,ZXFXQ,ZXSXJ]
    loss_shape = None

    label_cval = 255


    class_weight = None

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))#tf.ConfigProto一般用在创建session的时候。用来对session进行参数配置
    #使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加，由于不会释放内存，所以会导致碎片
    session = tf.Session(config=config)
    K.set_session(session)#设置会话吧

    train(batch_size, epochs, lr_base, lr_power, weight_decay, classes, model_name, train_file_path, val_file_path,
          data_dir, label_dir, target_size=target_size, batchnorm_momentum=batchnorm_momentum,  loss_fn=loss_fn, metrics=metrics,  label_cval=label_cval)
