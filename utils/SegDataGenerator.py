# coding=utf-8
from keras.preprocessing.image import *
from keras.applications.imagenet_utils import preprocess_input
from keras import backend as K
from PIL import Image
import numpy as np
import os

class SegDirectoryIterator(Iterator):
    '''
    init用于生成文件名称列表，然后调用iterator类生成数个下标，然后根据这些下标来调用数据
    '''
    def __init__(self, file_path, seg_data_generator,
                 data_dir, label_dir,  classes,
                 target_size, color_mode,
                 data_format,batch_size, shuffle):
        if data_format == 'default':
            data_format = K.image_data_format()#使用自带的图像维度定义方式
        self.file_path = file_path#Segmentation/train.txt
        self.data_dir = data_dir#train
        self.label_dir = label_dir#label
        self.classes = classes#8
        self.seg_data_generator = seg_data_generator#<utils.SegDataGenerator.SegDataGenerator object at 0x7f67957bc278>
        self.target_size = tuple(target_size)#512*512
        self.color_mode = color_mode#grayscale
        self.data_format = data_format#channels_last
        self.nb_label_ch=1#label的层数

       #设置train的通道组织方式
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                    self.image_shape = self.target_size + (3,)#target.size就是图片大小啦
            else:
                    self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':#单层的灰度图像
                    self.image_shape = self.target_size + (1,)
            else:
                    self.image_shape = (1,) + self.target_size
        #设置label的组织方式
        if self.data_format == 'channels_last':
                self.label_shape = self.target_size + (self.nb_label_ch,)#这里是尚未进行one-hot编码的
        else:
                self.label_shape = (self.nb_label_ch,) + self.target_size

        # 构建文件列表
        self.data_files = []
        self.label_files = []
        fp = open(file_path)#打开file_path,Segmentation/train.txt
        lines = fp.readlines()#将里面的文件名称全读出来
        fp.close()
        self.nb_sample = len(lines)

        for line in lines:
            line = line.strip('\n')
            self.data_files.append(line)
            self.label_files.append(line)#将train里面的图片读进来
        super(SegDirectoryIterator, self).__init__(
            self.nb_sample, batch_size, shuffle, seed=1)
        #首先找到test的父类（比如是类A），然后把类test的对象self转换为类A的对象，
        # 然后“被转换”的类A对象调用自己的__init__函数
    def _get_batches_of_transformed_samples(self, index_array):
        """这是被iterator类调用的一个东东,用于根据索引返回一批数据"""
        current_batch_size = len(index_array)#这里的index_array其实是一个列表，例如batch_size等于4那么len就是4

        if self.target_size:
            batch_x = np.zeros((current_batch_size,) + self.image_shape)
            #print("在这里的size为" + str(batch_x.shape))（1,512,512,1）

            batch_y = np.zeros((current_batch_size,) + self.label_shape,
                                   dtype=int)#这就是标签值的组织结构啦
        # 根据下标去取数据
        for i, j in enumerate(index_array):
            data_file = self.data_files[j]
            label_file = self.label_files[j]
            img =Image.open(os.path.join(self.data_dir, data_file))
            #去datadir读取文件，也就是原始图片

            label_filepath = os.path.join(self.label_dir, label_file)
            label = Image.open(label_filepath)
            # 加载标签图片

            x = img_to_array(img, data_format=self.data_format)
            y = img_to_array(label, data_format=self.data_format).astype(int)
            #生成batch数组
            batch_x[i] = x
            batch_y[i] = y
        batch_x /= 255.0#将数据缩放到0-1空间
        return batch_x, batch_y

class SegDataGenerator(object):
    #完成了一个工作：设置通道顺序
    def __init__(self):
        data_format = K.image_data_format()#看图片维度顺序
        self.data_format = data_format
    def flow_from_directory(self, file_path, data_dir,
                            label_dir,  classes,
                            target_size, color_mode,
                            batch_size, shuffle):
        return SegDirectoryIterator(
            file_path, self,
            data_dir=data_dir, label_dir=label_dir,
            classes=classes,
            target_size=target_size, color_mode=color_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle,)

