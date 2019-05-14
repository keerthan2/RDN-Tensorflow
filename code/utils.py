import tensorflow as tf
from tensorflow.python.layers.core import Dense
import numpy as np

wt_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
wt_reg = None

def batch_norm(X,is_training,i):
    output = tf.contrib.layers.batch_norm(X,
                              updates_collections=None,
                              decay=0.9,
                              center=True,
                              scale=True,
                              is_training=is_training,
                              trainable=is_training,
                              scope='BN_'+str(i),
                              reuse=tf.AUTO_REUSE,
                              fused=True,
                              zero_debias_moving_mean=True,
                              adjustment = lambda shape: ( tf.random_uniform(shape[-1:], 0.93, 1.07), tf.random_uniform(shape[-1:], -0.1, 0.1)),
                              renorm=False)
    return output


def data_augment(image_rgb, image_y):
    opt = tf.random_uniform((),minval=0,maxval=4,dtype=tf.int32)
    image_rgb = tf.cond(tf.equal(opt,0), lambda: tf.image.rot90(image_rgb,k=1), lambda: image_rgb)
    image_rgb = tf.cond(tf.equal(opt,1), lambda: tf.image.rot90(image_rgb,k=2), lambda: image_rgb)
    image_rgb = tf.cond(tf.equal(opt,2), lambda: tf.image.rot90(image_rgb,k=3), lambda: image_rgb)
    image_y = tf.cond(tf.equal(opt,0), lambda: tf.image.rot90(image_y,k=1), lambda: image_y)
    image_y = tf.cond(tf.equal(opt,1), lambda: tf.image.rot90(image_y,k=2), lambda: image_y)
    image_y = tf.cond(tf.equal(opt,2), lambda: tf.image.rot90(image_y,k=3), lambda: image_y)
    opt = tf.random_uniform((),minval=0,maxval=2,dtype=tf.int32)
    image_rgb = tf.cond(tf.equal(opt,0), lambda: tf.image.flip_left_right(image_rgb), lambda: image_rgb)
    image_y = tf.cond(tf.equal(opt,0), lambda: tf.image.flip_left_right(image_y), lambda: image_y)
    return image_rgb, image_y
        
def DownSample(x, h, scale=4):
    #ds_x = x.get_shape()
    #x = tf.reshape(x, [ds_x[0]*ds_x[1], ds_x[2], ds_x[3], 3])
    W = tf.constant(h)
    filter_height, filter_width = 13, 13
    pad_height = filter_height - 1
    pad_width = filter_width - 1
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_array = [[0,0], [pad_top, pad_bottom], [pad_left, pad_right], [0,0]]    
    depthwise_F = tf.tile(W, [1, 1, 3, 1])
    y = tf.nn.depthwise_conv2d(tf.pad(x, pad_array, mode='REFLECT'), depthwise_F, [1, scale, scale, 1], 'VALID')
    ds_y = y.get_shape()
    #y = tf.reshape(y, [ds_x[0], ds_x[1], ds_y[1], ds_y[2], 3])
    return y

def gkern(kernlen=13, nsig=1.6):
    import scipy.ndimage.filters as fi
    inp = np.zeros((kernlen, kernlen))
    inp[kernlen//2, kernlen//2] = 1
    return fi.gaussian_filter(inp, nsig)

def rdb(x,filters,i):
    inp =x
    for j in range(1,33):
        output = tf.layers.conv2d(inp,filters=64,kernel_size=3,strides=(1, 1),padding='same', activation=None,use_bias=False,name='RDB_'+str(i)+'conv_'+str(j))
        output = tf.nn.relu(output)
        inp = inp + output
    output = tf.concat([x,output],axis = -1)
    output = tf.layers.conv2d(output,filters=64,kernel_size=1,strides=(1,1),padding='same',activation=None,use_bias=False,name='RDB_'+str(i))
    output = tf.math.add(output,x)
    return output



