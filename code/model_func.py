import tensorflow as tf
from utils import *
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell
from tensorflow.contrib import rnn

def primary_model(x, is_training=True):
    output = x
    
    output = tf.layers.conv2d(output,filters=64,kernel_size=3,strides=(1, 1),padding='same', activation=None,use_bias=False,name='Conv_1')
    output2 = tf.layers.conv2d(output,filters=64,kernel_size=3,strides=(1, 1),padding='same', activation=None,use_bias=False,name='Conv_2')

    output3 =output2
    for i in range(1,21):
        output2 = rdb(output2,64,i)
        output3 = tf.concat([output3,output2],axis=-1)

    output3 = tf.layers.conv2d(output3,filters=64,kernel_size=1,strides=(1, 1),padding='same', activation=None,use_bias=False,name='Conv_3')
    output3 = tf.layers.conv2d(output3,filters=64,kernel_size=3,strides=(1, 1),padding='same', activation=None,use_bias=False,name='Conv_4')

    output = tf.math.add(output3,output)    

    for i in range(1,3):
        output = tf.layers.conv2d_transpose(output,filters=64,kernel_size=3,strides=(2, 2),padding='same',use_bias=False,name='deconv_'+str(i))
        output = batch_norm(output,is_training,i)        
        output = tf.nn.relu(output)
    
    output = tf.layers.conv2d(output,filters=1,kernel_size=3,strides=(1, 1),padding='same', activation=None,use_bias=False,name='Conv_5')
    
    output = tf.nn.tanh(output)
    output += 1.0
    output /= 2.0
    return output
