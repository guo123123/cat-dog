import glob
from sklearn.utils import shuffle
import os
import tensorflow as tf
import cv2
import numpy as np
classes = ['dogs','cats']
validation_size = 0.2
img_size = 128
num_channels = 3

file_path1 = 'E:/Python/cat/train_image/1/*.jpg'#dog
file_path2 = 'E:/Python/cat/train_image/0/*.jpg'#cat
file_path3 = 'E:/Python/cat/test_image/1/*.jpg'#dog
file_path4 = 'E:/Python/cat/test_image/0/*.jpg'#cat
def generated(train):
    if train:
        img_file1 = glob.glob(file_path1)
        img_file2 = glob.glob(file_path2)
        n = 5000
    else:
        img_file1 = glob.glob(file_path3)
        img_file2 = glob.glob(file_path4)
        n = 250
    images = []
    labels = []
    ur = []
    for i  in range(n):
        if i %1000 ==0:
            print('The dog i is %d'%i)
        a = img_file1[i]
        image = cv2.imread(a)
        image = image.astype(np.float32)
        image = np.multiply(image,1.0/255.0)
        images.append(image)
        label = np.zeros(len(classes))
        label[0]=1.0
        labels.append(label)
        ur.append(a)
    for i  in range(n):
        if i %1000 ==0:
            print('The cat i is %d'%i)
        a = img_file2[i]
        image = cv2.imread(a)
        image = image.astype(np.float32)
        image = np.multiply(image,1.0/255.0)
        images.append(image)
        label = np.zeros(2)
        label[1] = 1.0
        labels.append(label)
        ur.append(a)
    images = np.array(images)
    labels = np.array(labels)
    #ur = np.array(ur)



    return images,labels

# def DataSet(images, labels, img_names, cls):
#     pass
# images, labels, img_names, cls=  generated()
# images, labels, img_names, cls= shuffle(images, labels, img_names, cls)
# validation_size = int(validation_size*images.shape[0])
# validation_images = images[:validation_size]
# validation_labels = labels[:validation_size]
# validation_img_names = img_names[:validation_size]
# validation_cls = cls = cls[:validation_size]
#
# train_images = images[validation_size:]
# train_labels = labels[validation_size:]
# train_img_names = img_names[validation_size:]
# train_cls = cls[validation_size:]

# train = Dataset(train_images,train_labels,train_img_names,train_cls)
# valid = Dataset(validation_images,validation_labels,validation_img_names,validation_cls)
#
# with tf.Session()as sess :
#     x = tf.placeholder(tf.float32,shape=[None,img_size,img_size,num_channels])
#     y_true = tf.placeholder(tf.float32,shape=[None,2])
#     y_true_cls = tf.argmax(y_true,dimension=1)
#
