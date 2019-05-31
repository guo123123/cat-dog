import time
import tensorflow as tf
import CAD_forward
import CAD_backward
import numpy as np
import generated
TEST_INTERVAL_SECS = 5
from sklearn.utils import shuffle

def test():
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [
            500,
            CAD_forward.IMAGE_SIZE,
            CAD_forward.IMAGE_SIZE,
            CAD_forward.NUM_CHANNELS
        ], name='x')
        y_ = tf.placeholder(tf.float32, [500, CAD_forward.OUTPUT_NODE])
        y = CAD_forward.forward(x, False)

        # ema = tf.train.ExponentialMovingAverage(CAD_backward.MOVING_AVERAGE_DECAY)
        # ema_restore = ema.variables_to_restore()
        # saver = tf.train.Saver(ema_restore)

        correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        while True:
            with tf.Session() as sess:
                    saver = tf.train.Saver()
                    ckpt = tf.train.get_checkpoint_state(CAD_backward.MODEL_SAVE_PATH)
                    saver.restore(sess, ckpt.model_checkpoint_path)
                # ckpt = tf.train.get_checkpoint_state(CAD_backward.MODEL_SAVE_PATH)
                # if ckpt and ckpt.model_checkpoint_path:
                    #saver.restore(sess, ckpt.model_checkpoint_path)
                    #global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    images, labels= generated.generated(False)
                    images, labels= shuffle(images, labels)
                    accuracy_score = sess.run(accuracy, feed_dict={x: images, y_:labels})
                    print('After training step,test accuracy = %g' % (accuracy_score))
            #     else:
            #         print('No checkpoint file found')
            #         return
            time.sleep(TEST_INTERVAL_SECS)


def main():

    test()


if __name__ == '__main__':
    main()
