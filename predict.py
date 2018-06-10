# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
import cv2
from PIL import Image
from keras.preprocessing.image import *
from keras.models import load_model
import keras.backend as K
from keras.applications.imagenet_utils import preprocess_input
from utils.metrics import *
from models import *

def save_file(im1,picturename):
    im = np.zeros([im1.shape[0], im1.shape[1], 1], dtype=int)
    im[:, :, 0] = im1
    im[im == 7] = 250
    im[im == 6] = 200
    im[im == 5] = 150
    im[im == 4] = 100
    im[im == 3] = 75
    im[im == 2] = 50
    im[im == 1] = 25
    cv2.imwrite("test/predictlabel/全心脏/"+picturename, im)
def save_file1(im1,picturename):
    im = np.zeros([im1.shape[0], im1.shape[1], 1], dtype=int)
    im[:, :, 0] = im1
    im[im == 7] = 250
    im[im == 6] = 0
    im[im == 5] = 0
    im[im == 4] = 0
    im[im == 3] = 0
    im[im == 2] = 0
    im[im == 1] = 0
    cv2.imwrite("test/predictlabel/肺动脉/"+picturename, im)
def save_file2(im1,picturename):
    im = np.zeros([im1.shape[0], im1.shape[1], 1], dtype=int)
    im[:, :, 0] = im1
    im[im == 7] = 0
    im[im == 6] = 250
    im[im == 5] = 0
    im[im == 4] = 0
    im[im == 3] = 0
    im[im == 2] = 0
    im[im == 1] = 0
    cv2.imwrite("test/predictlabel/升主动脉/"+picturename, im)
def save_file3(im1,picturename):
    im = np.zeros([im1.shape[0], im1.shape[1], 1], dtype=int)
    im[:, :, 0] = im1
    im[im == 7] = 0
    im[im == 6] = 0
    im[im == 5] = 250
    im[im == 4] = 0
    im[im == 3] = 0
    im[im == 2] = 0
    im[im == 1] = 0
    cv2.imwrite("test/predictlabel/右心室血腔/"+picturename, im)
def save_file4(im1,picturename):
    im = np.zeros([im1.shape[0], im1.shape[1], 1], dtype=int)
    im[:, :, 0] = im1
    im[im == 7] = 0
    im[im == 6] = 0
    im[im == 5] = 0
    im[im == 4] = 250
    im[im == 3] = 0
    im[im == 2] = 0
    im[im == 1] = 0
    cv2.imwrite("test/predictlabel/右心房血腔/"+picturename, im)
def save_file5(im1,picturename):
    im = np.zeros([im1.shape[0], im1.shape[1], 1], dtype=int)
    im[:, :, 0] = im1
    im[im == 7] = 0
    im[im == 6] = 0
    im[im == 5] = 0
    im[im == 4] = 0
    im[im == 3] = 250
    im[im == 2] = 0
    im[im == 1] = 0
    cv2.imwrite("test/predictlabel/左心室血腔/"+picturename, im)
def save_file6(im1,picturename):
    im = np.zeros([im1.shape[0], im1.shape[1], 1], dtype=int)
    im[:, :, 0] = im1
    im[im == 7] = 0
    im[im == 6] = 0
    im[im == 5] = 0
    im[im == 4] = 0
    im[im == 3] = 0
    im[im == 2] = 250
    im[im == 1] = 0
    cv2.imwrite("test/predictlabel/左心房血腔/"+picturename, im)
def save_file7(im1,picturename):
    im = np.zeros([im1.shape[0], im1.shape[1], 1], dtype=int)
    im[:, :, 0] = im1
    im[im == 7] = 0
    im[im == 6] = 0
    im[im == 5] = 0
    im[im == 4] = 0
    im[im == 3] = 0
    im[im == 2] = 0
    im[im == 1] = 250
    cv2.imwrite("test/predictlabel/左心室心肌/"+picturename, im)

def load_train_data(path,model_name,weight_file):
    imgs = os.listdir(path)
    print(len( imgs ))

    current_dir = os.path.dirname( os.path.realpath( __file__ ) )#recent dir
    save_path = os.path.join( current_dir, 'Models/' + model_name )
    checkpoint_path = os.path.join( save_path, weight_file )  # 加载网络权重
    batch_shape = (1,) + (512,512) + (1,)
    model = globals()[model_name](batch_shape=batch_shape, input_shape=(512, 512, 1))
    model.load_weights(checkpoint_path, by_name=True)
    total=0
    for img_num in imgs:
        total+=1
        img_num = img_num.strip( '\n' )
        print( '第%s张图片: %s' % (total, img_num) )  # 0
        image = Image.open( '%s/%s' % (path, img_num) )
        image = img_to_array( image )
        image = np.expand_dims( image, axis=0 )
        image /= 255
        result8 = model.predict( image, batch_size=1 )  # 八通道结果
        result = np.argmax( result8, axis=-1 ).astype( np.uint8 )
        save_file( result[0], img_num )
        save_file1( result[0], img_num )
        save_file2( result[0], img_num )
        save_file3( result[0], img_num )
        save_file4( result[0], img_num )
        save_file5( result[0], img_num )
        save_file6( result[0], img_num )
        save_file7( result[0], img_num )
def getarray(picturename):
    image = Image.open(picturename)
    im = img_to_array(image)
    im[im >= 1]=1
    return im
def comput_dice(im1,im2):
    im=im1*im2
    fenzi=np.sum(im)
    fenmu=np.sum(im1)+sum(im2)
    if fenmu==0:
        return 1
    else:
        return  2*fenzi/fenmu
def computedice(path):

    imgs = os.listdir( path )
    count=0
    d0=0
    d1=0
    d2=0
    d3=0
    d4=0
    d5=0
    d6=0
    d7=0
    for img_num in imgs:
        count+=1
        im1=getarray("test/predictlabel/肺动脉/"+img_num)
        im2=getarray("test/viewlabel/肺动脉/"+img_num)
        temp=comput_dice(im1, im2)
        d0 += temp
        #

        im1=getarray("test/predictlabel/升主动脉/"+img_num)
        im2=getarray("test/viewlabel/升主动脉/"+img_num)
        temp=comput_dice(im1, im2)
        d1+= temp
        #

        im1=getarray("test/predictlabel/右心室血腔/"+img_num)
        im2=getarray("test/viewlabel/右心室血腔/"+img_num)
        temp=comput_dice(im1, im2)
        d2 += temp
        #

        im1=getarray("test/predictlabel/右心房血腔/"+img_num)
        im2=getarray("test/viewlabel/右心房血腔/"+img_num)
        temp=comput_dice(im1, im2)
        d3 += temp
        #

        im1=getarray("test/predictlabel/左心室血腔/"+img_num)
        im2=getarray("test/viewlabel/左心室血腔/"+img_num)
        temp=comput_dice(im1, im2)
        d4 += temp
        #

        im1=getarray("test/predictlabel/左心房血腔/"+img_num)
        im2=getarray("test/viewlabel/左心房血腔/"+img_num)
        temp=comput_dice(im1, im2)
        d5 += temp
        #

        im1=getarray("test/predictlabel/左心室心肌/"+img_num)
        im2=getarray("test/viewlabel/左心室心肌/"+img_num)
        temp=comput_dice(im1, im2)
        d6 += temp

    d7=(d0+d1+d2+d3+d4+d5+d6)/7
    d7/=count
    d0/=count
    d1/= count
    d2/= count
    d3/= count
    d4/= count
    d5/= count
    d6/= count
    print("平均的dice： %s"%d7)
    print("肺动脉的dice： %s" % d0)
    print("升主动脉的dice： %s" % d1)
    print("右心室血腔的dice： %s" % d2)
    print("右心房血腔的dice： %s" % d3)
    print("左心室血腔的dice： %s" % d4)
    print("左心房血腔的dice： %s" % d5)
    print("左心室心肌的dice： %s"%d6)
def compute_duliang_dice(path,model_name,weight_file):
    imgs = os.listdir(path)#将train中的图像全部读取出来
    print(len( imgs ))

    current_dir = os.path.dirname( os.path.realpath( __file__ ) )#recent dir
    save_path = os.path.join( current_dir, 'Models/' + model_name )
    checkpoint_path = os.path.join( save_path, weight_file )  # 加载网络权重

    batch_shape=(len(imgs),) + (512,512) + (1,)
    batch_x =np.zeros( batch_shape)
    batch_y=np.zeros( batch_shape,dtype=int )

    model = globals()[model_name](batch_shape=batch_shape, input_shape=(512, 512, 1))
    model.load_weights(checkpoint_path, by_name=True)
    total=0
    for img_num in imgs:
        img_num = img_num.strip( '\n' )

        image = Image.open( '%s/%s' % (path, img_num) )
        image = img_to_array( image, data_format='channels_last')
        batch_x[total]=image

        label = Image.open("test/label/%s"%img_num )
        label = img_to_array( label, data_format='channels_last').astype( int )
        batch_y[total]=label

        total += 1
    batch_x /= 255.0
    pred = model.predict(batch_x, batch_size=len(imgs) )
    pred1 = np.argmax(pred, axis=-1 ).astype( np.uint8 )
    save_file( pred1[0], "jinjinweileceshi.png" )

    sess = tf.InteractiveSession()
    temp0=FDM(batch_y,pred)
    temp1 =SZDM(batch_y,pred )
    temp2 = YXSXQ(batch_y,pred )
    temp3 = YXFXQ(batch_y,pred )
    temp4 = ZXSXQ(batch_y,pred )
    temp5 =ZXFXQ(batch_y,pred )
    temp6 = ZXSXJ(batch_y,pred )
    temp7=temp0+temp1+temp2+temp3+temp4+temp5+temp6
    sess.run(temp7)


    print("肺动脉的dice： %s" % temp0.eval())

    print("升主动脉的dice： %s" % temp1.eval())

    print("右心室血腔的dice： %s" % temp2.eval())

    print("右心房血腔的dice： %s" % temp3.eval())

    print("左心室血腔的dice： %s" % temp4.eval())

    print("左心房血腔的dice： %s" % temp5.eval())

    print("左心室心肌的dice： %s" % temp6.eval())

"""
    
def compute_duliang_dice(path,model_name,weight_file):
    imgs = os.listdir(path)
    print(len( imgs ))

    current_dir = os.path.dirname( os.path.realpath( __file__ ) )#recent dir
    save_path = os.path.join( current_dir, 'Models/' + model_name )
    checkpoint_path = os.path.join( save_path, weight_file )  # 加载网络权重

    batch_shape = (1,) + (512,512) + (1,)
    model = globals()[model_name](batch_shape=batch_shape, input_shape=(512, 512, 3))
    model.load_weights(checkpoint_path, by_name=True)
    total=0
    d0=0
    d1=0
    d2=0
    d3=0
    d4=0
    d5=0
    d6=0
    d7=0
    for img_num in imgs:
        total+=1
        img_num = img_num.strip( '\n' )
        print( '第%s张图片: %s' % (total, img_num) )  # 0
        image = Image.open( '%s/%s' % (path, img_num) )
        image = img_to_array( image )
        image = np.expand_dims( image, axis=0 )
        image /= 255
        pred= model.predict( image, batch_size=1 )  # 八通道结果

        true = np.zeros( (1,512,512,1),dtype=int )  # 这就是标签值的组织结构啦
        label = Image.open("test/label/%s"%img_num )
        label = img_to_array( label, data_format='channels_last').astype( int )
        true[0]=label

        sess = tf.InteractiveSession()
        temp0=FDM(true, pred)
        temp1 =SZDM( true, pred )
        temp2 = YXSXQ( true, pred )
        temp3 = YXFXQ( true, pred )
        temp4 = ZXSXQ( true, pred )
        temp5 =ZXFXQ( true, pred )
        temp6 = ZXSXJ( true, pred )
        temp7=temp0+temp1+temp2+temp3+temp4+temp5+temp6
        sess.run(temp7)

        d0+= temp0.eval()
        print("肺动脉的dice： %s" % temp0.eval())
        d1+= temp1.eval()
        print("升主动脉的dice： %s" % temp1.eval())
        d2 += temp2.eval()
        print("右心室血腔的dice： %s" % temp2.eval())
        d3 += temp3.eval()
        print("右心房血腔的dice： %s" % temp3.eval())
        d4 += temp4.eval()
        print("左心室血腔的dice： %s" % temp4.eval())
        d5 += temp5.eval()
        print("左心房血腔的dice： %s" % temp5.eval())
        d6 += temp6.eval()
        print("左心室心肌的dice： %s" % temp6.eval())
    d7+=(d0+d1+d2+d3+d4+d5+d6)/7
    d7/=count
    print("平均的dice： %s"%d7)
    print("肺动脉的dice： %s" % d0)
    print("升主动脉的dice： %s" % d1)
    print("右心室血腔的dice： %s" % d2)
    print("右心房血腔的dice： %s" % d3)
    print("左心室血腔的dice： %s" % d4)
    print("左心房血腔的dice： %s" % d5)
    print("左心室心肌的dice： %s"%d6)
"""
if __name__ == '__main__':

    model_name = 'AtrousFCN_Resnet50_16s'
    weight_file = 'checkpoint_weights.hdf5'
    image_size = (512, 512)
    #
    print( "#############################" )
    print("计算根据度量精确度函数计算出来的dice")
    print( "#############################" )
    compute_duliang_dice("test/train/",model_name,weight_file)
    #也就是说load_train_data会对结果造成影响
    model_name = 'AtrousFCN_Resnet50_16s'
    weight_file = 'checkpoint_weights.hdf5'
    load_train_data( "test/train/", model_name, weight_file )  # 根据根路径产生预测用的图片
    print( "-----------------------------" )
    print( "-----------------------------" )
    print( "-----------------------------" )
    print("计算根据产生的图片计算出来的dice")
    print( "-----------------------------" )
    computedice("test/train/")#根据已经产生好的图片计算dice指数