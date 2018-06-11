# coding=utf-8
from keras.models import Model
from keras.layers import *
from keras.regularizers import l2
import tensorflow as tf

def conv_relu(nb_filter, nb_row, nb_col, subsample=(1, 1), border_mode='same', bias = True, w_decay = 0.01):
    def f(x):
        with tf.name_scope('conv_relu'):
            x = Conv2D(filters=nb_filter, kernel_size=(nb_row, nb_col), stride=subsample, use_bias=bias,
                                 kernel_initializer="he_normal", W_regularizer=l2(w_decay), border_mode=border_mode)(x)
            x = Activation("relu")(x)
        return x
    return f

def conv_bn(nb_filter, nb_row, nb_col, subsample=(1, 1), border_mode='same', bias = True, w_decay = 0.01):
    def f(x):
        with tf.name_scope('conv_bn'):
            x = Conv2D(filters=nb_filter, kernel_size=(nb_row, nb_col), stride=subsample, use_bias=bias,
                              kernel_initializer="he_normal", W_regularizer=l2(w_decay), border_mode=border_mode)(x)
            x = BatchNormalization(mode=0, axis=-1)(x)
        return x
    return f

def conv_bn_relu(nb_filter, nb_row, nb_col, subsample=(1, 1), border_mode='same', bias = True, w_decay = 0.01):
    def f(x):
        with tf.name_scope('conv_bn_relu'):
            x = Conv2D(filters=nb_filter, kernel_size=(nb_row, nb_col), stride=subsample, use_bias=bias,
                              kernel_initializer="he_normal", W_regularizer=l2(w_decay), border_mode=border_mode)(x)
            x = BatchNormalization(mode=0, axis=-1)(x)
            x = Activation("relu")(x)
        return x
    return f

def bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1), border_mode='same', bias = True, w_decay = 0.01):
    def f(x):
        with tf.name_scope('bn_relu_conv'):
            x = BatchNormalization(mode=0, axis=-1)(x)
            x = Activation("relu")(x)
            x = Conv2D(filters=nb_filter, kernel_size=(nb_row, nb_col), stride=subsample, use_bias=bias,
                              kernel_initializer="he_normal", W_regularizer=l2(w_decay), border_mode=border_mode)(x)
        return x
    return f

def atrous_conv_bn(nb_filter, nb_row, nb_col, atrous_rate=(2, 2), subsample=(1, 1), border_mode='same', bias = True, w_decay = 0.01):
    def f(x):
        with tf.name_scope('atrous_conv_bn'):
            x = Conv2D(filters=nb_filter, kernel_size=(nb_row, nb_col), dilation_rate=atrous_rate, stride=subsample, use_bias=bias,
                       kernel_initializer="he_normal", kernel_regularizer=l2(w_decay), padding=border_mode)(x)
            x = BatchNormalization(mode=0, axis=-1)(x)
        return x
    return f

def atrous_conv_bn_relu(nb_filter, nb_row, nb_col, atrous_rate=(2, 2), subsample=(1, 1), border_mode='same', bias = True, w_decay = 0.01):
    def f(x):
        with tf.name_scope('atrous_conv_bn_relu'):
            x = Conv2D(filters=nb_filter, kernel_size=(nb_row, nb_col), dilation_rate=atrous_rate, stride=subsample, use_bias=bias,
                       kernel_initializer="he_normal", kernel_regularizer=l2(w_decay), padding=border_mode)(x)
            x = BatchNormalization(mode=0, axis=-1)(x)
            x = Activation("relu")(x)
        return x
    return f
def __conv_block(ip, nb_filter, bottleneck=False, dropout_rate=None, weight_decay=1e-4):
    ''' Apply BatchNorm, Relu, 3x3 Conv2D, optional bottleneck block and dropout
    Args:
        ip: Input keras tensor
        nb_filter: number of filters
        bottleneck: add bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor with batch_norm, relu and convolution2d added (optional bottleneck)
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(ip)
    x = Activation('relu')(x)
    print("BN+RELU")
    if bottleneck:#如果是瓶颈层，那么会卷积到4倍于原通道大小，并且加上
        inter_channel = nb_filter * 4  # Obtained from https://github.com/liuzhuang13/DenseNet/blob/master/densenet.lua

        x = Conv2D(inter_channel, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
                   kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)
        print("进入瓶颈层--CONV+BN+RELU")
        print(x.shape)
        print("--------------------------")
    x = Conv2D(nb_filter, (3, 3), kernel_initializer='he_normal', padding='same', use_bias=False)(x)
    print("CONV")
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    print(x.shape)
    return x


def __dense_block(x, nb_layers, nb_filter, growth_rate, bottleneck=False, dropout_rate=None, weight_decay=1e-4,
                  grow_nb_filters=True, return_concat_list=False):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
    Args:
        x: keras tensor
        nb_layers: the number of layers of conv_block to append to the model.
        nb_filter: number of filters
        growth_rate: growth rate
        bottleneck: bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        grow_nb_filters: flag to decide to allow number of filters to grow
        return_concat_list: return the list of feature maps along with the actual output
    Returns: keras tensor with nb_layers of conv_block appended
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x_list = [x]

    for i in range(nb_layers):
        print("进入一个denseblock"+str(i))
        cb = __conv_block(x, growth_rate, bottleneck, dropout_rate, weight_decay)
        x_list.append(cb)

        x = concatenate([x, cb], axis=concat_axis)#将数组连接到一块，用来准备下一次卷积

        if grow_nb_filters:#看心情是否增加滤波器数量咯
            nb_filter += growth_rate

    if return_concat_list:
        return x, nb_filter, x_list
    else:
        return x, nb_filter


def __transition_block(ip, nb_filter, compression=1.0, weight_decay=1e-4):
    ''' Apply BatchNorm, Relu 1x1, Conv2D, optional compression, dropout and Maxpooling2D
    Args:
        ip: keras tensor
        nb_filter: number of filters
        compression: calculated as 1 - reduction. Reduces the number of feature maps
                    in the transition block.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(ip)
    x = Activation('relu')(x)
    x = Conv2D(int(nb_filter * compression), (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #BN加上RELU加上卷积，这里使用了池化来缩小
    print("denseblock之间连接:BN+RELU+CONV+MAXPOOL")

    return x
#(img_input,  nb_dense_block=5,growth_rate=12, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4,nb_layers_per_block=4, init_conv_filters=48, input_shape=None)
def create_fcn_dense_net( img_input,nb_dense_block=5, growth_rate=12,
                           reduction=0.0, dropout_rate=None, weight_decay=1e-4,
                           nb_layers_per_block=4,
                           init_conv_filters=48):
    ''' 建立DenseNet模型
    ARGS：
        nb_classes：类的数量
        img_input：形状元素（通道，行，列）或（行，列，通道）
        include_top：标志包含最终密集层
        nb_dense_block：要添加到结尾的密集块的数量（通常= 3）
        growth_rate：每个密集块添加的过滤器数量
        减少：过渡块的缩减因子。注意：缩小值被反转以计算压缩
        dropout_rate：辍学率
        weight_decay：L2正则化项
        nb_layers_per_block：每个密集块中的层数。
            可以是一个正整数或一个列表。
            如果是正整数，则为每个密集块设置一定数量的图层。
            如果列表，nb_layer按照提供的方式使用。请注意列表大小必须
            be（nb_dense_block + 1）
        nb_upsampling_conv：通过子像素卷积上采样中的卷积层数
        upsampling_type：可以是'upsampling'，'deconv'和'subpixel'之一。定义
            所使用的上采样算法的类型。
        input_shape：仅用于完全卷积网络中的形状推理。
        激活：顶层的激活类型。可以是'softmax'或'sigmoid'之一。
                    请注意，如果使用sigmoid，则类必须为1。
    返回：附加了conv_block的nb_layers的keras张量
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1


    nb_layers = list(nb_layers_per_block)  #
    assert len(nb_layers) == (nb_dense_block + 1), 'If list, nb_layer is used as provided. ' \
                                                       'Note that list size must be (nb_dense_block + 1)'
    bottleneck_nb_layers = nb_layers[-1]
    rev_layers = nb_layers[::-1]
    nb_layers.extend(rev_layers[1:])

    # compute compression factor
    compression = 1.0 - reduction

    # 初始模块
    x = Conv2D(init_conv_filters, (7, 7), kernel_initializer='he_normal', padding='same', name='initial_conv2D',
               use_bias=False, kernel_regularizer=l2(weight_decay))(img_input)
    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)

    print("CONV+BN+Relu")
    nb_filter = init_conv_filters#48

    skip_list = []

    # Add dense blocks and transition down block
    for block_idx in range(nb_dense_block):#0-5，也就是15不会被计算进去，这里的densenetblock指明了数量为5个denseblock
        x, nb_filter = __dense_block(x, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate,
                                     weight_decay=weight_decay)#[4, 5, 7, 10, 12, 15]，这下子知道每一个多大了
        #构建一个densenet块
        #注意4, 5, 7, 10, 12个卷积块，并且这里并没有出现任何瓶颈层
        # Skip connection，准备用作下一次的连接
        skip_list.append(x)

        # transition_block，这里使用了平均池化来减小大小，用于在密集快之间的连接
        x = __transition_block(x, nb_filter, compression=compression, weight_decay=weight_decay)

        nb_filter = int(nb_filter * compression)  # this is calculated inside transition_down_block

    # 进行最后一次的卷积,这里是15层
    _, nb_filter, concat_list = __dense_block(x, bottleneck_nb_layers, nb_filter, growth_rate,
                                              dropout_rate=dropout_rate, weight_decay=weight_decay,
                                              return_concat_list=True)
    """
    skip_list = skip_list[::-1]  #对跳跃连接进行反序，也就是15个小块

    # Add dense blocks and transition up block
    for block_idx in range(nb_dense_block):
        n_filters_keep = growth_rate * nb_layers[nb_dense_block + block_idx]#滤波器数量

        # not the concatenation of the input with the feature maps (concat_list[0].
        #concat_list是指最后一个densenet块的所有子卷积层
        l = concatenate(concat_list[1:], axis=concat_axis)

        #进行一个上采样操作
        t = __transition_up_block(l, nb_filters=n_filters_keep, type=upsampling_type, weight_decay=weight_decay)

        # 和原来的对应位置贴在一块
        x = concatenate([t, skip_list[block_idx]], axis=concat_axis)

        # 重新生成densenetblock
        x_up, nb_filter, concat_list = __dense_block(x, nb_layers[nb_dense_block + block_idx + 1], nb_filter=growth_rate,
                                                 growth_rate=growth_rate, dropout_rate=dropout_rate,
                                                 weight_decay=weight_decay, return_concat_list=True,
                                                 grow_nb_filters=False)
        print("上采样完毕："+str(x_up.shape))

    if include_top:
        x = Conv2D(nb_classes, (1, 1), activation='linear', padding='same', use_bias=False)(x_up)

        if K.image_data_format() == 'channels_first':
            channel, row, col = input_shape
        else:
            row, col, channel = input_shape

        x = Reshape((row * col, nb_classes))(x)
        x = Activation(activation)(x)
        x = Reshape((row, col, nb_classes))(x)
    else:
        x = x_up

    """
    return x

def __transition_up_block(ip, nb_filters, type='deconv', weight_decay=1E-4):
    ''' SubpixelConvolutional Upscaling (factor = 2)
    Args:
        ip: keras tensor
        nb_filters: number of layers
        type: can be 'upsampling', 'subpixel', 'deconv'. Determines type of upsampling performed
        weight_decay: weight decay factor
    Returns: keras tensor, after applying upsampling operation.
    '''


    x = UpSampling2D()(ip)#进行上采样

    return x
