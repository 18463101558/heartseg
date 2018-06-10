#该文件用来读取一张png
#coding:utf-8
import numpy as np
from PIL import Image
rootpath=""
path="2007_000175.png"
img=Image.open(rootpath+path)
im=np.array(img)
print(im.shape)
im[im==7]=250
im[im==6]=200
im[im==5]=150
im[im==4]=100
im[im==3]=75
im[im==2]=50
im[im==1]=25
img.show()

