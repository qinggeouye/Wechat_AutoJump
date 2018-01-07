import tensorflow as tf
import numpy as np


class JumpModelFine:
    def __init__(self):
        self.img_shape = (320, 320)
        self.batch_size = 8
        self.input_channel = 3
        self.out_channel = 2

    @staticmethod
    def conv2d(name, in_para, ks, stride):
        with tf.name_scope(name):
            with tf.variable_scope(name):
                w = tf.get_variable('%s-w' % name, shape=ks, initializer=tf.truncated_normal_initializer())
                b = tf.get_variable('%s-b' % name, shape=[ks[-1]], initializer=tf.constant_initializer())
                out_para = tf.nn.conv2d(in_para, w, strides=[1, stride, stride, 1], padding='SAME',
                                        name='%s-conv' % name)
                out_para = tf.nn.bias_add(out_para, b, name='%s-bias_add' % name)
        return out_para

    def make_conv_bn_relu(self, name, in_para, ks, stride, is_training):
        out_para = self.conv2d('%s-conv' % name, in_para, ks, stride)
        out_para = tf.layers.batch_normalization(out_para, name='%s-bn' % name, training=is_training)
        out_para = tf.nn.relu(out_para, name='%s-relu' % name)
        return out_para

    @staticmethod
    def make_fc(name, in_para, ks):
        with tf.name_scope(name):
            with tf.variable_scope(name):
                w = tf.get_variable('%s-w' % name, shape=ks, initializer=tf.truncated_normal_initializer())
                b = tf.get_variable('%s-b' % name, shape=[ks[-1]], initializer=tf.constant_initializer())
                out_para = tf.matmul(in_para, w, name='%s-mat' % name)
                out_para = tf.nn.bias_add(out_para, b, name='%s-bias_add' % name)
                # out_para = tf.nn.dropout(out_para, keep_prob, name='%s-drop' % name)
        return out_para

    def forward(self, img, is_training, name='fine'):
        # print(name)
        with tf.name_scope(name):
            with tf.variable_scope(name):
                out_para = self.conv2d('conv1', img, [3, 3, self.input_channel, 16], 2)
                # out_para = tf.layers.batch_normalization(out_para, name='bn1', training=is_training)
                out_para = tf.nn.relu(out_para, name='relu1')

                out_para = self.make_conv_bn_relu('conv2', out_para, [3, 3, 16, 64], 1, is_training)
                out_para = tf.nn.max_pool(out_para, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

                out_para = self.make_conv_bn_relu('conv3', out_para, [5, 5, 64, 128], 1, is_training)
                out_para = tf.nn.max_pool(out_para, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

                out_para = self.make_conv_bn_relu('conv4', out_para, [7, 7, 128, 256], 1, is_training)
                out_para = tf.nn.max_pool(out_para, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

                out_para = self.make_conv_bn_relu('conv5', out_para, [9, 9, 256, 512], 1, is_training)
                out_para = tf.nn.max_pool(out_para, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

                out_para = tf.reshape(out_para, [-1, 512 * 10 * 10])
                out_para = self.make_fc('fc1', out_para, [512 * 10 * 10, 512])
                out_para = self.make_fc('fc2', out_para, [512, 2])

        return out_para


if __name__ == '__main__':
    model = JumpModelFine()
    # print(tf.zeros((1, 320, 320, 3)))
    # 第二个参数bool类型化
    out = model.forward(tf.zeros((1, 320, 320, 3)), tf.placeholder(np.bool, name='is_training'))
    print(out.get_shape().as_list())
