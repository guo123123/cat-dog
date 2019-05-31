import tensorflow as tf
import CAD_forward
import numpy as np
import  os
import generated
from sklearn.utils import shuffle
from tensorflow.examples.tutorials.mnist import input_data
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.0001
LEARNING_RATE_DECAY=0.99
#REGULARIZER = 0.0001
STEPS =50000
MOVING_AVERAGE_DECAY = 0.9
MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'CAD_model'
BATCH_SIZE = 100
def backward():
    print('it is backward')
    #卷积输入要求4阶张量
    x = tf.placeholder(tf.float32,[
    100,
    CAD_forward.IMAGE_SIZE,
    CAD_forward.IMAGE_SIZE,
    CAD_forward.NUM_CHANNELS
    ])
    y_true= tf.placeholder(tf.float32,[100,2])
    y = CAD_forward.forward(x,True)

    # global_step = tf.Variable(0,trainable=False)
    # ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.arg_max(y_,1))
    # cem = tf.reduce_mean(ce)
    # loss = cem + tf.add_n(tf.get_collection('losses'))

    # learning_rate = tf.train.exponential_decay(
    #     LEARNING_RATE_BASE,
    #     global_step,
    #     50000,
    #     LEARNING_RATE_DECAY,
    #     staircase=True
    # )
    y_pred = tf.nn.softmax(y)
    cross = tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_true)
    loss = tf.reduce_mean(cross)
    #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step)
    train_step = tf.train.AdamOptimizer(0.000001).minimize(loss)

    # ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    # ema_op = ema.apply(tf.trainable_variables())

    # with tf.control_dependencies([train_step,ema_op]):
    #     train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
        images, labels = generated.generated(True)
        images, labels = shuffle(images, labels)
        for j in range(300):

            print('轮数 is %d'%j)

            for i in range(9):
                train_images = images[i*100:(i+1)*100]
                train_labels = labels[i*100:(i+1)*100]

                sess.run(train_step,feed_dict={x:train_images,y_true:train_labels})

            loss_value = sess.run(loss,feed_dict={x:train_images,y_true:train_labels})
            print('After %d training steps,loss on training batch is %g.'%(j,loss_value))
            saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=49900+j)

def main():
    #mnist =input_data.read_data_sets('./datas/MNIST_data',one_hot=True)
    backward()
if __name__  =='__main__':
       main()
