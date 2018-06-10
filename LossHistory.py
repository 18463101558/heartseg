
# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Mon Feb  5 15:31:25 2018

@author: brucelau
"""

'''Trains a simple deep NN on the MNIST dataset.'''

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import LambdaCallback
import numpy as np


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        print("第一批训练开始！")

    def on_epoch_end(self, epoch, logs={}):
        train = open("logs/losshistory.txt", 'a+')
        epoch = epoch
        loss = logs['loss']
        val_loss = logs['val_loss']

        dice_acc= logs['average']
        val_dice_acc= logs['val_average']

        record = str(epoch) + "," + str(loss) + "," + str(dice_acc) + ","\
                 + str(val_loss) + ","+str(val_dice_acc)
        train.write(record + "\n")
        train.close()

        #FDM,SZDM,YXSXQ,YXFXQ,ZXSXQ,ZXFXQ,ZXSXJ,background,average
        train = open("logs/acchistory.txt", 'a+')
        average = logs['average']
        val_average = logs['val_average']

        SZDM= logs['SZDM']
        val_SZDM= logs['val_SZDM']

        YXSXQ= logs['YXSXQ']
        val_YXSXQ= logs['val_YXSXQ']

        YXFXQ= logs['YXFXQ']
        val_YXFXQ= logs['val_YXFXQ']

        ZXSXJ = logs['ZXSXJ']
        val_ZXSXJ = logs['val_ZXSXJ']

        ZXSXQ= logs['ZXSXQ']
        val_ZXSXQ= logs['val_ZXSXQ']

        ZXFXQ= logs['ZXFXQ']
        val_ZXFXQ= logs['val_ZXFXQ']

        FDM = logs['FDM']
        val_FDM = logs['val_FDM']

        record =  str(FDM) + "," + str(val_FDM) + ","\
                 + str(SZDM) + ","+ str(val_SZDM) + ","+str(YXSXQ) + "," + str(val_YXSXQ)+ ","\
                 + str(YXFXQ) + "," + str(val_YXFXQ) + "," + str(ZXSXQ) + "," + str(val_ZXSXQ)+ "," \
                 + str(ZXFXQ) + "," + str(val_ZXFXQ) + "," + str(ZXSXJ) + "," + str(val_ZXSXJ)+ ","\
                 +str(average) + "," + str(val_average)
        train.write(record + "\n")
        train.close()






