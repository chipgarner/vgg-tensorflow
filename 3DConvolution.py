import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize
import Paths
from PIL import Image

class Conv3D:
    def __init__(self, img):

        images = tf.stack([img, img, img, img, img, img, img, img, img])
        images = tf.reshape(images, [1, 9, 34, 54, 1])
        images = tf.to_float(images, "input")

        assert images.get_shape() == [1, 9, 34, 54, 1]

        self.inputs = images

        self.out = self.build_phase1(tf.placeholder(tf.float32, [1, 9, 34, 54, 1]))

    def build_phase1(self, inputs):
        with tf.name_scope("phase1"):
            c1 = tf.layers.conv3d(inputs, filters=7, kernel_size=[5, 7, 7], name='C1')
            r1 = tf.abs(c1, name='R1')
            s1 = tf.layers.max_pooling3d(r1, pool_size=1, strides=[1, 2, 2], name='S1')
            c2 = tf.layers.conv3d(s1, filters=35, kernel_size=[3, 5, 5], name='C2')
            r2 = tf.abs(c2, name='R2')
            s2 = tf.layers.max_pooling3d(r2, pool_size=1, strides=[1, 2, 2], name='S2')
            c3 = tf.layers.conv3d(s2, filters=5, kernel_size=[3, 3, 3], name='C3')
            c3_flat = tf.reshape(c3, [-1, 5*3*8*1])
            n1 = tf.layers.dense(inputs=c3_flat, units=50, name='N1')
            self.n2 = tf.layers.dense(n1, units=6, name='N2')
            return self.n2 # logits


if __name__ == '__main__':
    directory = Paths.this_directory()

    # # convert('L') is gray scale
    # img = Image.open(directory + '/Tests/Images/laska.png').convert('L')
    # img.show()
    # img = img.resize((34, 54))
    # img.show()
    # processed_image = img.getdata()

    img = imread(directory + '/Tests/Images/laska.png', mode='L') #L for gray scale
    img = imresize(img, (34, 54))
    image = Image.fromarray(img, 'L')
    image.show()

    out = Conv3D(img)

    assert out.out.shape == (1, 6)

    with tf.Session() as sess:
        prob = sess.run(out.inputs)

        writer = tf.summary.FileWriter("output", sess.graph)
        writer.close()
