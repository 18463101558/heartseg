#coding:utf-8
#此函数用于将未标记图像转换到0-255的空间
import nibabel as nib
import numpy as np
import cv2
import os
#定义train的数据目录
import random
#返回全体文件列表名称
def file_name(file_dir):
    filelist=[]
    for (*rest, files)in os.walk(file_dir):
        #filelist.append(files)
        return files

#根据文件名称保存文件
def save_file(filepath,filename,savepath):
    nimg = nib.load(filepath+"".join(filename))#读取文件数据
    im = np.array(nimg.get_data())
    maxcount = np.amax(im)#找出当前矩阵最大值，注意这
    im = im * 255.0 / 3136#因为测试得知图像最大像素为3136，所以这里可以将其归一化到
    print(u"ct训练图像处理完毕一个"+str(maxcount))
    n = nimg.shape[-1]#找出一共有多少层
    filename="".join(filename)
    filename=filename.split('.')[0]#去掉后缀的nii.gz之类的东东
    for i in range(n):
        processim = np.zeros([nimg.shape[0], nimg.shape[1], 1], dtype=int)
        processim[:, :, 0] = im[:, :, i]
        new_path = savepath + "/"+filename + str(i) + '.png'
        cv2.imwrite(new_path, processim)
def save_label_file(filepath,filename,savepath):
    nimg = nib.load(filepath+"".join(filename))
    im = np.array(nimg.get_data())
    maxcount = np.amax(im)
    im[im == 850] = 7
    im[im == 820] = 6
    im[im == 600] = 5
    im[im == 550] = 4
    im[im == 500] = 3
    im[im == 420] = 2
    im[im == 205] = 1
    print(u"ct标签图像处理最大值为：" + str(maxcount))

    n = nimg.shape[-1]#找出一共有多少层
    filename="".join(filename)#转化成str的格式
    filename=filename.split('.')[0]#因为是tar.gz的形式，故后缀名字被丢弃
    for i in range(n):
        processim = np.zeros([nimg.shape[0], nimg.shape[1], 1], dtype=int)
        processim[:, :, 0] = im[:, :, i]
        new_path = savepath + "/" + filename + str(i) + '.png'
        cv2.imwrite(new_path, processim)

def produce_val_and_train(file_dir):#从train文件夹中选择20%作为验证集,方法是通过遍历train文件夹得到的

    train = open("Segmentation/train.txt", 'w')
    val=open("Segmentation/val.txt", 'w')
    i=0
    for *rest, files in os.walk(file_dir+ "/"):
        for filename in files:
            i=i+1  # 产生1-5的随机数
            if i%5==0:
                val.write(filename + "\n")
            else:
                train.write(filename + "\n")
    train.close()
    val.close()

def view_label_file1(filepath,filename,savepath):
    nimg = nib.load(filepath+"".join(filename))
    im = np.array(nimg.get_data())
    maxcount = np.amax(im)
    im[im == 850] = 250
    im[im == 820] = 0
    im[im == 600] = 0
    im[im == 550] = 0
    im[im == 500] = 0
    im[im == 420] = 0
    im[im == 205] = 0
    print(u"肺动脉图像处理完毕：" + str(maxcount))

    n = nimg.shape[-1]#找出一共有多少层
    filename="".join(filename)#转化成str的格式
    filename=filename.split('.')[0]#因为是tar.gz的形式，故后缀名字被丢弃
    for i in range(n):
        processim = np.zeros([nimg.shape[0], nimg.shape[1], 1], dtype=int)
        processim[:, :, 0] = im[:, :, i]
        new_path = savepath + "/" + filename + str(i) + '.png'
        cv2.imwrite(new_path, processim)
def view_label_file2(filepath,filename,savepath):
    nimg = nib.load(filepath+"".join(filename))
    im = np.array(nimg.get_data())
    maxcount = np.amax(im)
    im[im == 850] = 0
    im[im == 820] = 250
    im[im == 600] = 0
    im[im == 550] = 0
    im[im == 500] = 0
    im[im == 420] = 0
    im[im == 205] = 0
    print(u"升主动脉图像处理完毕：" + str(maxcount))

    n = nimg.shape[-1]#找出一共有多少层
    filename="".join(filename)#转化成str的格式
    filename=filename.split('.')[0]#因为是tar.gz的形式，故后缀名字被丢弃
    for i in range(n):
        processim = np.zeros([nimg.shape[0], nimg.shape[1], 1], dtype=int)
        processim[:, :, 0] = im[:, :, i]
        new_path = savepath + "/" + filename + str(i) + '.png'
        cv2.imwrite(new_path, processim)
def view_label_file3(filepath,filename,savepath):
    nimg = nib.load(filepath+"".join(filename))
    im = np.array(nimg.get_data())
    maxcount = np.amax(im)
    im[im == 850] = 0
    im[im == 820] = 0
    im[im == 600] = 250
    im[im == 550] = 0
    im[im == 500] = 0
    im[im == 420] = 0
    im[im == 205] = 0
    print(u"右心室血腔图像处理完毕：" + str(maxcount))

    n = nimg.shape[-1]#找出一共有多少层
    filename="".join(filename)#转化成str的格式
    filename=filename.split('.')[0]#因为是tar.gz的形式，故后缀名字被丢弃
    for i in range(n):
        processim = np.zeros([nimg.shape[0], nimg.shape[1], 1], dtype=int)
        processim[:, :, 0] = im[:, :, i]
        new_path = savepath + "/" + filename + str(i) + '.png'
        cv2.imwrite(new_path, processim)
def view_label_file4(filepath,filename,savepath):
    nimg = nib.load(filepath+"".join(filename))
    im = np.array(nimg.get_data())
    maxcount = np.amax(im)
    im[im == 850] = 0
    im[im == 820] = 0
    im[im == 600] = 0
    im[im == 550] = 250
    im[im == 500] = 0
    im[im == 420] = 0
    im[im == 205] = 0
    print(u"右心房血腔图像处理完毕：" + str(maxcount))

    n = nimg.shape[-1]#找出一共有多少层
    filename="".join(filename)#转化成str的格式
    filename=filename.split('.')[0]#因为是tar.gz的形式，故后缀名字被丢弃
    for i in range(n):
        processim = np.zeros([nimg.shape[0], nimg.shape[1], 1], dtype=int)
        processim[:, :, 0] = im[:, :, i]
        new_path = savepath + "/" + filename + str(i) + '.png'
        cv2.imwrite(new_path, processim)
def view_label_file5(filepath,filename,savepath):
    nimg = nib.load(filepath+"".join(filename))
    im = np.array(nimg.get_data())
    maxcount = np.amax(im)
    im[im == 850] = 0
    im[im == 820] = 0
    im[im == 600] = 0
    im[im == 550] = 0
    im[im == 500] = 250
    im[im == 420] = 0
    im[im == 205] = 0
    print(u"左心室血腔图像处理完毕：" + str(maxcount))

    n = nimg.shape[-1]#找出一共有多少层
    filename="".join(filename)#转化成str的格式
    filename=filename.split('.')[0]#因为是tar.gz的形式，故后缀名字被丢弃
    for i in range(n):
        processim = np.zeros([nimg.shape[0], nimg.shape[1], 1], dtype=int)
        processim[:, :, 0] = im[:, :, i]
        new_path = savepath + "/" + filename + str(i) + '.png'
        cv2.imwrite(new_path, processim)


def view_label_file6(filepath, filename, savepath):
    nimg = nib.load( filepath + "".join( filename ) )
    im = np.array( nimg.get_data() )
    maxcount = np.amax( im )
    im[im == 850] = 0
    im[im == 820] = 0
    im[im == 600] = 0
    im[im == 550] = 0
    im[im == 500] = 0
    im[im == 420] = 250
    im[im == 205] = 0
    print(u"左心房血腔图像处理完毕：" + str( maxcount ))

    n = nimg.shape[-1]  # 找出一共有多少层
    filename = "".join( filename )  # 转化成str的格式
    filename = filename.split( '.' )[0]  # 因为是tar.gz的形式，故后缀名字被丢弃
    for i in range( n ):
        processim = np.zeros( [nimg.shape[0], nimg.shape[1], 1], dtype=int )
        processim[:, :, 0] = im[:, :, i]
        new_path = savepath + "/" + filename + str( i ) + '.png'
        cv2.imwrite( new_path, processim )


def view_label_file7(filepath, filename, savepath):
    nimg = nib.load( filepath + "".join( filename ) )
    im = np.array( nimg.get_data() )
    maxcount = np.amax( im )
    im[im == 850] = 0
    im[im == 820] = 0
    im[im == 600] = 0
    im[im == 550] = 0
    im[im == 500] = 0
    im[im == 420] = 0
    im[im == 205] = 250
    print(u"左心室心肌图像处理完毕：" + str( maxcount ))

    n = nimg.shape[-1]  # 找出一共有多少层
    filename = "".join( filename )  # 转化成str的格式
    filename = filename.split( '.' )[0]  # 因为是tar.gz的形式，故后缀名字被丢弃
    for i in range( n ):
        processim = np.zeros( [nimg.shape[0], nimg.shape[1], 1], dtype=int )
        processim[:, :, 0] = im[:, :, i]
        new_path = savepath + "/" + filename + str( i ) + '.png'
        cv2.imwrite( new_path, processim )

def view_label_file(filepath, filename, savepath):
    nimg = nib.load( filepath + "".join( filename ) )
    im = np.array( nimg.get_data() )
    maxcount = np.amax( im )
    im[im == 850] = 250
    im[im == 820] = 200
    im[im == 600] = 150
    im[im == 550] = 100
    im[im == 500] = 75
    im[im == 420] = 50
    im[im == 205] = 25
    print(u"全心脏图像处理完毕：" + str( maxcount ))

    n = nimg.shape[-1]  # 找出一共有多少层
    filename = "".join( filename )  # 转化成str的格式
    filename = filename.split( '.' )[0]  # 因为是tar.gz的形式，故后缀名字被丢弃
    for i in range( n ):
        processim = np.zeros( [nimg.shape[0], nimg.shape[1], 1], dtype=int )
        processim[:, :, 0] = im[:, :, i]
        new_path = savepath + "/" + filename + str( i ) + '.png'
        cv2.imwrite( new_path, processim )
if __name__ == '__main__':
    # trainrootpath = "origindata/ct_train/"
    # trainsavepath = "train"
    # trainpath=file_name(trainrootpath)#取出该文件夹目录下面所有文件
    # for filename in trainpath:
    #     save_file(trainrootpath,filename, trainsavepath)
    #
    # labelrootpath = "origindata/ct_label/"
    # labelsavepath = "label"
    # labelpath=file_name(labelrootpath)
    # for filename in labelpath:
    #     save_label_file(labelrootpath,filename, labelsavepath)
    # #到前面的所有步骤是为了将ct图像分成png
    # #产生训练集和验证集的文件列表
    # produce_val_and_train(trainsavepath)

    # labelrootpath = "origindata/ct_label/"
    # labelsavepath = "viewlabel/肺动脉/"
    # labelpath=file_name(labelrootpath)
    # for filename in labelpath:
    #     view_label_file1(labelrootpath,filename, labelsavepath)
    #
    #
    # labelrootpath = "origindata/ct_label/"
    # labelsavepath = "viewlabel/升主动脉/"
    # labelpath=file_name(labelrootpath)
    # for filename in labelpath:
    #     view_label_file2(labelrootpath,filename, labelsavepath)
    #
    # labelrootpath = "origindata/ct_label/"
    # labelsavepath = "viewlabel/右心室血腔/"
    # labelpath=file_name(labelrootpath)
    # for filename in labelpath:
    #     view_label_file3(labelrootpath,filename, labelsavepath)
    #
    # labelrootpath = "origindata/ct_label/"
    # labelsavepath = "viewlabel/右心房血腔/"
    # labelpath=file_name(labelrootpath)
    # for filename in labelpath:
    #     view_label_file4(labelrootpath,filename, labelsavepath)
    #
    # labelrootpath = "origindata/ct_label/"
    # labelsavepath = "viewlabel/左心室血腔/"
    # labelpath=file_name(labelrootpath)
    # for filename in labelpath:
    #     view_label_file5(labelrootpath,filename, labelsavepath)
    #
    # labelrootpath = "origindata/ct_label/"
    # labelsavepath = "viewlabel/左心房血腔/"
    # labelpath=file_name(labelrootpath)
    # for filename in labelpath:
    #     view_label_file6(labelrootpath,filename, labelsavepath)
    #
    # labelrootpath = "origindata/ct_label/"
    # labelsavepath = "viewlabel/左心室心肌/"
    # labelpath=file_name(labelrootpath)
    # for filename in labelpath:
    #     view_label_file7(labelrootpath,filename, labelsavepath)

    labelrootpath = "origindata/ct_label/"
    labelsavepath = "viewlabel/全心脏/"
    labelpath=file_name(labelrootpath)ls
    for filename in labelpath:
        view_label_file(labelrootpath,filename, labelsavepath)