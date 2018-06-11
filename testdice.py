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

#今晚的目的是检验度量函数是否写错啦
#首先写出预测图片，看看预测结果的dice系数
#再根据预测图片one-hot编码，看看dice系数
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
    cv2.imwrite("predictlabel/全心脏/"+picturename, im)
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
    cv2.imwrite("predictlabel/肺动脉/"+picturename, im)
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
    cv2.imwrite("predictlabel/升主动脉/"+picturename, im)
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
    cv2.imwrite("predictlabel/右心室血腔/"+picturename, im)
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
    cv2.imwrite("predictlabel/右心房血腔/"+picturename, im)
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
    cv2.imwrite("predictlabel/左心室血腔/"+picturename, im)
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
    cv2.imwrite("predictlabel/左心房血腔/"+picturename, im)
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
    cv2.imwrite("predictlabel/左心室心肌/"+picturename, im)
def inference(model_name, weight_file, image_size, image_list, data_dir, label_dir, return_results=True, save_dir=None):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    print(current_dir)#这玩意就是获取当前目录
    batch_shape = (1, ) + image_size + (1, )
    save_path = os.path.join(current_dir, 'Models/'+model_name)
    checkpoint_path = os.path.join(save_path, weight_file)#加载网络权重

    #######################################配置参数#############################
    #config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    #session = tf.Session(config=config)
    #K.set_session(session)
    ######################################加载网络##############################
    model = globals()[model_name](batch_shape=batch_shape, input_shape=(512, 512, 1))
    model.load_weights(checkpoint_path, by_name=True)
    total = 0
    for img_num in image_list:
        img_num = img_num.strip('\n')
        total += 1
        print('第%s张图片: %s' % (total,img_num))#0
        image = Image.open('%s/%s' % (data_dir, img_num))
        image = img_to_array(image)  # , data_format='default')
        image = np.expand_dims(image, axis=0)
        image/=255
        result8 = model.predict(image, batch_size=1)#八通道结果
        result = np.argmax(result8, axis=-1).astype(np.uint8)
        save_file(result[0],img_num)
        save_file1(result[0], img_num)
        save_file2(result[0], img_num)
        save_file3(result[0], img_num)
        save_file4(result[0], img_num)
        save_file5(result[0], img_num)
        save_file6(result[0], img_num)
        save_file7(result[0], img_num)
        print(result.shape)
        return result8
def load_train_data():
    imgs = glob.glob("test/predict/*." + png )
    print(len( imgs ))
    imgdatas = np.ndarray( (len( imgs ), self.out_rows, self.out_cols, 8), dtype=np.uint8 )
    imglabels = np.ndarray( (len( imgs ), self.out_rows, self.out_cols, 1), dtype=np.uint8 )
    for imgname in imgs:
        midname = imgname[imgname.rindex( "/" ) + 1:]
        img = load_img( self.data_path + "/" + midname, grayscale=True )
        label = load_img( self.label_path + "/" + midname, grayscale=True )
        im = img_to_array( img )
        im[im == 250] =7
        im[im == 200] =6
        im[im == 150] =5
        im[im == 100] =4
        im[im == 75] =3
        im[im == 50] =2
        im[im == 25] =1
        im=categorical_crossentropy()
        label = img_to_array(label)
        imgdatas[i] = img
        imglabels[i] = label
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format( i, len( imgs ) ))
        i += 1
    print('loading done')
def getarray(picturename):
    image = Image.open(picturename)
    im = img_to_array(image)
    im[im >= 1]=1
    return im
def get_one_hot_arrey(picturename):
    image = Image.open(picturename)
    im = img_to_array(image)
    im[im == 250] = 7
    im[im == 200] = 6
    im[im == 150] = 5
    im[im == 100] = 4
    im[im == 75] = 3
    im[im == 50] = 2
    im[im == 25] = 1
    return im
def comput_dice(im1,im2):
    im=im1*im2
    fenzi=np.sum(im)
    fenmu=np.sum(im1)+sum(im2)
    if fenmu==0:
        return 1
    else:
        return  2*fenzi/fenmu
def compute_origin_dice():

    im1=getarray("predictlabel/肺动脉/1001tr200.png")
    im2=getarray("viewlabel/肺动脉/1001tr200.png")
    d0=comput_dice(im1, im2)
    print("肺动脉的dice： %s"%d0)

    im1=getarray("predictlabel/升主动脉/1001tr200.png")
    im2=getarray("viewlabel/升主动脉/1001tr200.png")
    d1=comput_dice(im1, im2)
    print("升主动脉的dice： %s"%d1)

    im1=getarray("predictlabel/右心室血腔/1001tr200.png")
    im2=getarray("viewlabel/右心室血腔/1001tr200.png")
    d2=comput_dice(im1, im2)
    print("右心室血腔的dice： %s"%d2)

    im1=getarray("predictlabel/右心房血腔/1001tr200.png")
    im2=getarray("viewlabel/右心房血腔/1001tr200.png")
    d3=comput_dice(im1, im2)
    print("右心房血腔的dice： %s"%d3)

    im1=getarray("predictlabel/左心室血腔/1001tr200.png")
    im2=getarray("viewlabel/左心室血腔/1001tr200.png")
    d4=comput_dice(im1, im2)
    print("左心室血腔的dice： %s"%d4)

    im1=getarray("predictlabel/左心房血腔/1001tr200.png")
    im2=getarray("viewlabel/左心房血腔/1001tr200.png")
    d5=comput_dice(im1, im2)
    print("左心房血腔的dice： %s"%d5)

    im1=getarray("predictlabel/左心室心肌/1001tr200.png")
    im2=getarray("viewlabel/左心室心肌/1001tr200.png")
    d6=comput_dice(im1, im2)
    print("左心室心肌的dice： %s"%d6)

    d7=(d0+d1+d2+d3+d4+d5+d6)/7
    print("平均的dice： %s"%d7)
def savepicture(name,im2):
    im = np.zeros([im2.shape[0], im2.shape[1]], dtype=int)
    im[:, :] = im2
    im[im == 7] = 250
    im[im == 6] = 200
    im[im == 5] = 150
    im[im == 4] = 100
    im[im == 3] = 75
    im[im == 2] = 50
    im[im == 1] = 25
    cv2.imwrite(name, im)

def compute_predict_dice(result):
    im1= get_one_hot_arrey("viewlabel/全心脏/1001tr200.png")  # 获取尚未one-hot编码的内容
    y_true = np.zeros((1, 512, 512, 1))
    y_true[0] = im1
    y_pred=result
    sess = tf.InteractiveSession()
    nb_classes = y_pred.shape[-1]   # 获取类别数量
    y_pred = K.reshape(y_pred, (-1, nb_classes))  # 压缩成二维,其实感觉好像也没啥必要来着
    y_pred= tf.nn.softmax( y_pred )#经历一个softmax将其转变成二维
    y_true=K.one_hot(tf.to_int32(K.flatten(y_true)),
                   nb_classes)
    y_pred=K.flatten(y_pred)
    y_true=K.flatten(y_true)
    fenzi=K.sum(2*tf.to_float(np.multiply(y_pred,y_true)))
    fenmu=K.sum(tf.to_float(y_pred))+K.sum(tf.to_float(y_true))
    result= 1-fenzi/fenmu
    sess.run(result)
    print(result.eval())

    # y_true = get_one_hot_arrey("viewlabel/全心脏/1001tr200.png")  # 获取尚未one-hot编码的内容
    # im1 = np.zeros((1, 512, 512, 1))
    # im1[0] = y_true
    #
    # sess = tf.InteractiveSession()
    # feidongmai = FDM(im1,result)
    # shengzhudongmai=SZDM(im1,result)
    # youxinshixueqiang=YXSXQ(im1,result)
    # youxinfangxueqiang=YXFXQ(im1,result)
    # zuoxinshixueqiang=ZXSXQ(im1,result)#这一个出毛病了
    # zuoxinfangxueqiang=ZXFXQ(im1,result)
    # zuoxinshixinji=ZXSXJ(im1,result)
    # #dice_background=background(im1,result)
    # average =(feidongmai + shengzhudongmai + youxinshixueqiang + \
    #           youxinfangxueqiang + zuoxinshixueqiang + zuoxinfangxueqiang + zuoxinshixinji)/7 #+ dice_background
    # sess.run(average)
    # print("肺动脉的dice： %s" % feidongmai.eval())
    # print("升主动脉的dice： %s" % shengzhudongmai.eval())
    # print("右心室血腔的dice： %s" % youxinshixueqiang.eval())
    # print("右心房血腔的dice： %s" % youxinfangxueqiang.eval())
    # print("左心室血腔的dice： %s" % zuoxinshixueqiang.eval())
    # print("左心房血腔的dice： %s" % zuoxinfangxueqiang.eval())
    # print("左心室心肌的dice： %s" % zuoxinshixinji.eval())
    # print("平均的dice： %s" % average.eval())
if __name__ == '__main__':

    model_name = 'AtrousFCN_Resnet50_16s'
    weight_file = 'checkpoint_weights.hdf5'
    image_size = (512, 512)
    data_dir        = os.path.expanduser('train')
    label_dir       = os.path.expanduser('label')
    image_list=["1001tr200.png"]
    #将所有文件目录下面的文件可视化并且予以保存
    result=inference(model_name, weight_file, image_size, image_list, data_dir, label_dir)

    compute_origin_dice()
    compute_predict_dice(result)