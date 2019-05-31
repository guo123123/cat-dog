import tensorflow as tf
import os,glob
import sys,argparse
import numpy as np
import cv2
import CAD_forward
def destore_model():
    print('2')
    images = []
    image = cv2.imread('7.jpg')
    image = cv2.resize(image,(128,128),0,0,cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images,dtype=np.int8)
    images = images.astype('float32')
    images = np.multiply(images,1.0/255.0)
    x_batch = images.reshape(1,128,128,3)
    graph = tf.get_default_graph()
    x = tf.placeholder(tf.float32, [
        1,
        128,
        128,
        3
    ])
    y_ = CAD_forward.forward(x, False, None)
    y_test_images = np.zeros((1,2))
    sess = tf.Session()
    saver = tf.train.import_meta_graph('./model/mnist_model-542.ckpt.meta')
    saver.restore(sess, './model/mnist_model-542.ckpt')
    feed_dict_testing = {x:x_batch,y_:y_test_images}
    result = sess.run(y_,feed_dict=feed_dict_testing)
    res_label = ['cat','dog']
    print(res_label[result.argmax()])
destore_model()