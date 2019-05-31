import tensorflow as tf
import numpy as np
IMAGE_SIZE =128
NUM_CHANNELS=3
CONV1_SIZE = 3   #第一层卷积核大小
CONV1_KERNEL_NUM = 32  #第一层使用了32个卷积核
CONV2_SIZE = 3
CONV2_KERNEL_NUM = 64
OUTPUT_NODE = 2   #10分类输出
FC_SIZE = 1024  #隐藏层节点个数
#def get_weight(shape,regularizer):
    # w = tf.Variable(tf.truncated_normal(shape,stddev=0.05))
    # if regularizer!=None:
    #     tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
    # return w
def get_weight(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.05))
def get_bias(shape):
    b = tf.Variable(tf.constant(0.05,shape=shape))
    return b
#求卷积
def conv2d(x,w): #x 输入，所用卷积核W

    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def forward(x,train):
    #初始化化第一层卷积核W ，B
    #conv1_w = get_weight([CONV1_SIZE,CONV1_SIZE,3,CONV1_KERNEL_NUM],regularizer)
    conv1_w = get_weight([CONV1_SIZE, CONV1_SIZE, 3, CONV1_KERNEL_NUM])
    conv1_b = get_bias([CONV1_KERNEL_NUM])
    conv1 = conv2d(x,conv1_w)
    #对conv1添加偏执，使用relu激活函数
    conv1 += conv1_b
    #relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_b))
    relu1 = tf.nn.relu(conv1)
    #池化
    pool1 = max_pool_2x2(relu1)
    #64*64*32

    #conv2_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV1_KERNEL_NUM], regularizer)
    conv2_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV1_KERNEL_NUM])
    conv2_b = get_bias([CONV1_KERNEL_NUM])
    conv2 = conv2d(pool1, conv2_w)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    pool2= max_pool_2x2(relu2)#第二层卷积的输出
    print(pool2.shape)
    #32*32*32
    #conv3_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV1_KERNEL_NUM], regularizer)
    conv3_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV1_KERNEL_NUM])
    conv3_b = get_bias([CONV1_KERNEL_NUM])
    conv3 = conv2d(pool2, conv3_w)
    relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_b))
    pool3 = max_pool_2x2(relu3)  # 第三层卷积的输出
    print(pool3.shape)
    #16*16*32
    #conv4_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM], regularizer)
    conv4_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM])
    conv4_b = get_bias([CONV2_KERNEL_NUM])
    conv4 = conv2d(pool3, conv4_w)
    relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_b))
    pool4 = max_pool_2x2(relu4)  # 第四层卷积的输出
    #8*8*64

    print(pool4.shape)
    pool_shape = pool4.get_shape().as_list()#得到pool4 输出矩阵的维度，存入list中

    #提取特征的长，宽，深度
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    print(nodes)
    print(pool_shape[0])
    #pool_shape[0]一个batch的值
    #将pool2 表示成，pool_shape[0]行，nodes列
    print(pool_shape)
    reshaped = tf.reshape(pool4,[pool_shape[0],nodes])
    # 全连接网络
    #第一层
    fc1_w = get_weight([nodes,FC_SIZE])
    fc1_b = get_bias([FC_SIZE])
    fc1 =tf.matmul(reshaped,fc1_w)+fc1_b
    if train:fc1 = tf.nn.dropout(fc1,0.7)
    fc1 = tf.nn.relu(fc1)
     #第二层
    fc2_w = get_weight([FC_SIZE,OUTPUT_NODE])
    fc2_b = get_bias([OUTPUT_NODE])
    fc2 = tf.matmul(fc1, fc2_w) + fc2_b
    if train: fc2 = tf.nn.dropout(fc2, 0.7)

    return fc2

