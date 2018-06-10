#coding:utf-8
import numpy as np
from PIL import Image
import keras.backend as K
import tensorflow as tf

K.sum(tf.to_float( K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1))))
im = K.argmax(y_true, axis=-1)
sess = tf.Session()
result = sess.run(im)
sess = tf.Sessi