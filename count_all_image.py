# -*- coding: utf-8 -*-
#此文件用于统计所有图片灰度像素级别数量

import os

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
pixel_count= np.zeros((1000))#初始化为全零的一维数组,用于统计像素数量
#print pixel_count

def read_one_png(i,my_file_path,name="/ct_label_1001_label"):
    img = Image.open(my_file_path + name+str(i)+".png")
    im = np.array(img)
    arr = im.flatten()  # 折叠成一维数组
    #print(count)证明count是一个全局变量
    for i in range(0, len(arr)):
        pixel_count[arr[i]] = pixel_count[arr[i]]+1

def read_png_image(my_file_path):
    count = 0
    ls = os.listdir(my_file_path)
    for i in ls:
        if os.path.isfile(os.path.join(my_file_path, i)):
            count += 1
    print( u"该文件夹目录下面一共有" +str(count)+ "张png图像")
    for i in range(0,count):#如果count是363，则i为0-362
        read_one_png(i,my_file_path)
        print (u"开始处理第"+str(i)+ u"张png图像")
    for j in range(0, len(pixel_count)):
        if(pixel_count[j]!=0):
            print (u"像素大小为"+str(j)+u"数量为"+str(pixel_count[j]))
if __name__ == '__main__':
    rootpath = ("/home/ye/桌面/我的tensorflow之旅/heart/")
    path = "ct_label"
    read_png_image(rootpath+path)

