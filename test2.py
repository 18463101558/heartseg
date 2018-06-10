#coding:utf-8
import numpy as np
from PIL import Image
import keras.backend as K
import tensorflow as tf
def show(im):
    im[im == 7] = 250
    # im[im == 6] = 200
    # im[im == 5] = 150
    # im[im == 4] = 100
    # im[im == 3] = 75
    # im[im == 2] = 50
    # im[im == 1] = 25
    im = Image.fromarray(np.uint8(im))
    im.save("label2.png")
def show2(img):
    y_true = K.one_hot(img, 8)
    im=K.argmax(y_true, axis=-1)
    sess = tf.Session()
    result = sess.run(im)
    show(result)
if __name__ == '__main__':

    path="label/1001tr200.png"
    im=Image.open(path)
    im=np.array(im)
    show2(im)
